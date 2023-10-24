import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from time import time

from model.InverseNetwork import InverseNetworkQuadChannel, inverseNetworkTrainingQuadChannel
from myutils import make_dir
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import pandas as pd
import random
class QCCPN(nn.Module):
    def __init__(self, n_class=20, gamma=0.1, feature_dim=240, input_feature=100,
                 distance_func=None,
                 ):

        super(QCCPN, self).__init__()

        self.num_class = n_class
        self.num_features = feature_dim
        self.gamma = gamma

        if distance_func is not None:
            self.distance_func = distance_func
        else:
            self.distance_func = lambda x: torch.sqrt(torch.sum(x ** 2, 1))

        self.feature = nn.Linear(input_feature, self.num_features)
        self.feature_rotate = nn.Linear(input_feature, self.num_features)

        self.centers = torch.nn.Parameter(torch.randn(self.num_class, self.num_features*4))
        self.centers.requires_grad = True

    def forward(self, x, x_90, x_180, x_270, return_fea=False):
        x_all = self.forward_feature(x, x_90, x_180, x_270)
        distances = self.distance_centers(x_all) * -1 * self.gamma
        if return_fea:
            return distances, x_all
        return distances


    def split_distance_centers_nobatch(self, x, labels, d):
        center = self.centers[labels]
        center = center.view(d, -1)
        x = self.forward_feature(torch.unsqueeze(x, 0))
        x = x.view(d, -1)
        dis = self.distance_func((x-center))
        return dis

    def forward_feature(self, x, x_90, x_180, x_270):
        # x = self.backbone(x)
        x = F.relu(self.feature(x))

        # x_90 = self.backbone(x_90)
        x_90 = F.relu(self.feature_rotate(x_90))

        # x_180 = self.backbone(x_180)
        x_180 = F.relu(self.feature_rotate(x_180))

        # x_270 = self.backbone(x_270)
        x_270 = F.relu(self.feature_rotate(x_270))

        x_all = torch.cat((x, x_90, x_180, x_270), 1)

        return x_all

    def distance_centers(self, features):
        num_class = self.centers.size(0)
        batch_size = features.size(0)
        expand_features = features.repeat_interleave(num_class, dim=0)
        expand_centers = self.centers.repeat(batch_size, 1)
        x = self.distance_func((expand_features - expand_centers))
        x = x.view(batch_size, num_class)
        return x

    def calc_distance(self, features, others):
        num_others = others.size(0)
        batch_size = features.size(0)
        expand_features = features.repeat_interleave(num_others, dim=0)
        expand_centers = others.repeat(batch_size, 1)
        x = self.distance_func((expand_features - expand_centers))
        x = x.view(batch_size, num_others)
        return x

    def extend_prototypes(self, n_class):
        print("n_class", n_class)
        old_num_class = self.num_class
        self.num_class += n_class

        newcenters = torch.nn.Parameter(torch.randn(self.num_class, self.num_features * 4))

        with torch.no_grad():
            newcenters[:old_num_class] = self.centers

        self.centers = newcenters
        print("new centers shape", self.centers.shape)

class QCCPNLoss(nn.Module):

    def __init__(self, gamma, epsilon_contrast, dis_func=None, device=None, w1=1.0):
        super(QCCPNLoss, self).__init__()
        self.gamma = gamma
        self.epsilon_contrast = epsilon_contrast
        self.w1 = w1
        self.dis_func = lambda x: torch.sum((x) ** 2, 1) if dis_func is None else dis_func
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") if device is None else device

    def forward(self, features, labels, centers):
        distance = self.distance_centers(features, centers)
        loss_dce = self.cross_entropy(self.gamma * -distance, labels)
        lcontrast = self.distance_contrast_batch(labels, centers)
        l2 = torch.clip(self.epsilon_contrast - lcontrast, min=0)
        l2 = torch.mean(l2)
        return loss_dce + self.w1 * l2

    def distance_centers(self, features, centers):
        num_class = centers.size(0)
        batch_size = features.size(0)

        expand_features = features.repeat_interleave(num_class, dim=0)
        expand_centers = centers.repeat(batch_size, 1)

        x = self.dis_func(expand_features - expand_centers)
        x = x.view(batch_size, num_class)
        return x
    def distance_contrast_batch(self, labels, centers):
        labels_size = labels.size(0)
        centers_size = centers.size(0)

        labels_centers = centers[labels]

        expand_labels = labels_centers.repeat_interleave(centers_size - 1, dim=0)
        expand_centers = centers.repeat(labels_size, 1)

        idx_label = torch.tensor(list(range(labels.size(0)))).to(self.device)
        idx_label = centers_size * idx_label + labels
        idx_select = torch.tensor(list(range(expand_centers.size(0)))).to(self.device)
        mask_idx_select = (idx_select != idx_label[0])

        for i in range(1, idx_label.size(0)):
            mask_idx_select &= (idx_select != idx_label[i])
        mask_idx_select = torch.nonzero(mask_idx_select).squeeze()
        expand_centers = torch.index_select(expand_centers, 0, mask_idx_select)

        x = self.dis_func(expand_labels - expand_centers)
        x = x.view(labels_size, centers_size - 1)
        x = torch.mean(x, 1)
        return x

class ClassIncrementalQCCPNLossOLD(QCCPNLoss):
    def __init__(self, gamma, epsilon_contrast, dis_func=None, device=None, w1=1.0, w2=1.0, uthreshold=0.95):
        super(ClassIncrementalQCCPNLossOLD, self).__init__(gamma, epsilon_contrast, dis_func=dis_func, device=device, w1=w1)
        self.w2 = w2
        # self.uthreshold = torch.tensor(uthreshold).to(device)
        self.uthreshold = uthreshold
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, labels, centers, ufea):
        distance = self.distance_centers(features, centers)
        loss_dce = self.cross_entropy(self.gamma * -distance, labels)
        lcontrast = self.distance_contrast_batch(labels, centers)
        l2 = torch.clip(self.epsilon_contrast - lcontrast, min=0)
        l2 = torch.mean(l2)
        distance_ufea = self.distance_centers(ufea, centers)
        softmax_ufea = self.softmax(distance_ufea).data
        max_ufea, _ = torch.max(softmax_ufea, dim=1)
        l3 = torch.clip(max_ufea - self.uthreshold, min=0)
        l3 = torch.mean(l3)
        return loss_dce + self.w1 * l2 + self.w2 * l3

class ClassIncrementalQCCPNLoss(QCCPNLoss):
    def __init__(self, gamma, epsilon_contrast, dis_func=None, device=None, w1=1.0, w2=1.0):
        super(ClassIncrementalQCCPNLoss, self).__init__(gamma, epsilon_contrast, dis_func=dis_func, device=device, w1=w1)
        self.w2 = w2
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, labels, centers, ufea):
        distance = self.distance_centers(features, centers)
        loss_dce = self.cross_entropy(self.gamma * -distance, labels)
        lcontrast = self.distance_contrast_batch(labels, centers)
        l2 = torch.clip(self.epsilon_contrast - lcontrast, min=0)
        l2 = torch.mean(l2)

        distance_ufea = self.distance_centers(ufea, centers)
        l3 = torch.clip(self.epsilon_contrast - distance_ufea, min=0)
        l3 = torch.mean(l3)
        return loss_dce + self.w1 * l2 + self.w2 * l3

def evalQCCPN(device, model, task_id, test_task_dl, unknown_dl, verbose, return_threshold=False):
    total = 0
    correct = 0
    prob_known = torch.Tensor([]).to(device)
    label_known = torch.Tensor([]).to(device)
    prob_unknown = torch.Tensor([]).to(device)
    label_unknown = torch.Tensor([]).to(device)
    predicted_known = torch.Tensor([]).to(device)
    predicted_unknown = torch.Tensor([]).to(device)
    with torch.no_grad():
        for t in range(task_id+1):
            for fea, fea90, fea180, fea270, labels, task in test_task_dl[t]:
                fea = fea.to(device)
                fea90 = fea90.to(device)
                fea180 = fea180.to(device)
                fea270 = fea270.to(device)
                labels = labels.to(device)

                outputs = model(fea, fea90, fea180, fea270)
                label_known = torch.cat((label_known, torch.ones(labels.size(0)).to(device)), 0)
                prob, predicted = torch.max(nn.functional.softmax(outputs, dim=1).data, 1)
                prob_known = torch.cat((prob_known, prob), 0)
                predicted_known = torch.cat((predicted_known, predicted), 0)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        for fea, fea90, fea180, fea270, labels, _ in unknown_dl:
            fea = fea.to(device)
            fea90 = fea90.to(device)
            fea180 = fea180.to(device)
            fea270 = fea270.to(device)
            labels = labels.to(device)
            outputs = model(fea, fea90, fea180, fea270)

            label_unknown = torch.cat((label_unknown, torch.zeros(labels.size(0)).to(device)), 0)
            prob, predicted = torch.max(nn.functional.softmax(outputs, dim=1).data, 1)
            predicted_unknown = torch.cat((predicted_unknown, predicted), 0)
            prob_unknown = torch.cat((prob_unknown, prob), 0)

        out_pred = torch.cat((prob_known, prob_unknown), 0).detach().cpu().numpy()
        out_label = torch.cat((label_known, label_unknown), 0).detach().cpu().numpy()
        fpr, tpr, threshold = sklearn.metrics.roc_curve(y_true=out_label, y_score=out_pred, pos_label=1)
        roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'j-index': tpr - fpr, 'threshold': threshold})
        # roc_df.to_csv('logs/figures/roc_detail_proposed.csv')
        roc_df_maxj = roc_df.sort_values('j-index', ascending=False)
        threshold = roc_df_maxj.iloc[0]['threshold']
        auroc = sklearn.metrics.auc(fpr, tpr)

        predicted_known = predicted_known.where(prob_known >= threshold, torch.tensor(0).to(device))
        predicted_known = predicted_known.where(prob_known < threshold, torch.tensor(1).to(device))
        predicted_unknown = predicted_unknown.where(prob_unknown >= threshold, torch.tensor(0).to(device))
        predicted_unknown = predicted_unknown.where(prob_unknown < threshold, torch.tensor(1).to(device))
        predicted_baccu = torch.cat((predicted_known, predicted_unknown), 0).detach().cpu().numpy()
        baccu = balanced_accuracy_score(out_label, predicted_baccu)

    if return_threshold:
        return 100 * correct / total, auroc, baccu, threshold

    return 100 * correct / total, auroc, baccu

def evalBACCUQCCPN(device, threshold, model, task_id, test_task_dl, unknown_dl, verbose):
    total = 0
    correct = 0
    prob_known = torch.Tensor([]).to(device)
    label_known = torch.Tensor([]).to(device)
    prob_unknown = torch.Tensor([]).to(device)
    label_unknown = torch.Tensor([]).to(device)
    predicted_known = torch.Tensor([]).to(device)
    predicted_unknown = torch.Tensor([]).to(device)
    with torch.no_grad():
        for t in range(task_id+1):
            for fea, fea90, fea180, fea270, labels, task in test_task_dl[t]:
                fea = fea.to(device)
                fea90 = fea90.to(device)
                fea180 = fea180.to(device)
                fea270 = fea270.to(device)
                labels = labels.to(device)

                outputs = model(fea, fea90, fea180, fea270)
                label_known = torch.cat((label_known, torch.ones(labels.size(0)).to(device)), 0)
                prob, predicted = torch.max(nn.functional.softmax(outputs, dim=1).data, 1)
                prob_known = torch.cat((prob_known, prob), 0)
                predicted_known = torch.cat((predicted_known, predicted), 0)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        for fea, fea90, fea180, fea270, labels, _ in unknown_dl:
            fea = fea.to(device)
            fea90 = fea90.to(device)
            fea180 = fea180.to(device)
            fea270 = fea270.to(device)
            labels = labels.to(device)
            outputs = model(fea, fea90, fea180, fea270)

            label_unknown = torch.cat((label_unknown, torch.zeros(labels.size(0)).to(device)), 0)
            prob, predicted = torch.max(nn.functional.softmax(outputs, dim=1).data, 1)
            predicted_unknown = torch.cat((predicted_unknown, predicted), 0)
            prob_unknown = torch.cat((prob_unknown, prob), 0)

        # out_pred = torch.cat((prob_known, prob_unknown), 0).detach().cpu().numpy()
        out_label = torch.cat((label_known, label_unknown), 0).detach().cpu().numpy()

        predicted_known = predicted_known.where(prob_known >= threshold, torch.tensor(0).to(device))
        predicted_known = predicted_known.where(prob_known < threshold, torch.tensor(1).to(device))
        predicted_unknown = predicted_unknown.where(prob_unknown >= threshold, torch.tensor(0).to(device))
        predicted_unknown = predicted_unknown.where(prob_unknown < threshold, torch.tensor(1).to(device))
        predicted_baccu = torch.cat((predicted_known, predicted_unknown), 0).detach().cpu().numpy()
        baccu = balanced_accuracy_score(out_label, predicted_baccu)

    return 100 * correct / total, baccu

def QCCPNTraining(device, model,  save_model_path, log_path, train_task_dl, test_task_dl, unknown_dl, feature_memory=None, load_path=None, learning_rate=0.001, start_epoch=0, n_epoch=50, batch_size=32,
                  gamma=0.1, contrastive=100, trial=5, initial_seed=1, return_model=False, save_last_model=False):
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

    make_dir(save_model_path)


    start_time = time()
    model.train()
    print("DEVICE:", device)
    model.to(device)
    n_epoch += start_epoch
    n_task = len(train_task_dl)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # default gamma=0.1, contrastive=1000
    qccpn_loss = QCCPNLoss(gamma, contrastive, device=device)
    acc_pertask = []
    auroc_pertask = []
    baccu_pertask = []
    task_desc_list = []
    task_id_list = []
    current_seed = initial_seed
    for seed_trial in range(trial):
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)
        seed = current_seed

        for task_id in range(n_task):
            total_step = len(train_task_dl[task_id])

            if task_id > 0:
                model.extend_prototypes(n_class_per_task)
                model.to(device)
                if feature_memory:
                    feature_memory.adjust(list(range(0, end_class)), batch_size)

            model.train()
            for epoch in range(start_epoch, n_epoch):
                total_loss = 0
                loss_step = 0
                total = 0
                correct = 0
                for i, (fea, fea90, fea180, fea270, labels, task) in enumerate(train_task_dl[task_id]):
                    fea = fea.to(device)
                    fea90 = fea90.to(device)
                    fea180 = fea180.to(device)
                    fea270 = fea270.to(device)
                    labels = labels.to(device)

                    if task_id > 0 and feature_memory:
                        # TODO check for QCCPN
                        lat_fea_mem, lat90_fea_mem, lat180_fea_mem, lat270_fea_mem, lbl_fea_mem, task_mem = feature_memory.take()
                        fea = torch.cat((fea, lat_fea_mem.to(device)), 0)
                        fea90 = torch.cat((fea90, lat90_fea_mem.to(device)), 0)
                        fea180 = torch.cat((fea180, lat180_fea_mem.to(device)), 0)
                        fea270 = torch.cat((fea270, lat270_fea_mem.to(device)), 0)
                        labels = torch.cat((labels, lbl_fea_mem.to(device)), 0)

                    output, features = model(fea, fea90, fea180, fea270, return_fea=True)

                    total += labels.size(0)
                    prob, predicted = torch.max(nn.functional.softmax(output, dim=1).data, 1)
                    correct += (predicted == labels).sum().item()

                    loss = qccpn_loss(features, labels, model.centers)

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    loss_step += 1

                print(
                    f"TASK: {task_id} - {task[0]} | Epoch [{epoch + 1}/{n_epoch}]: Loss: {total_loss / loss_step}, Accuracy: {100 * correct / total}")

            task_desc_list.append(task[0])
            task_id_list.append(task_id)
            start_class = int(task[0].split(',')[0].split(':')[1].lstrip().split("(")[-1])
            end_class = int(task[0].split(',')[1].lstrip().split(')')[0])
            n_class_per_task = end_class - start_class
            # print("n_class_pertask", n_class_per_task)

            model.to(device)

            model.eval()
            test_avg_acc, auroc, baccu = evalQCCPN(device, model, task_id, test_task_dl, unknown_dl, verbose=1)
            acc_pertask.append(test_avg_acc)
            auroc_pertask.append(auroc)
            baccu_pertask.append(baccu)
            print(f"Test Average Acc from first task 0 to current task {task_id}: {test_avg_acc}")
            print(f"AUROC from first task 0 to current task {task_id}: {auroc}")
            print(f"BACCU from first task 0 to current task {task_id}: {baccu}")

        current_seed +=1

    endtime = time()

    print(f"Total running time: {endtime-start_time}")
    logdf = pd.DataFrame(
        {"task id": task_id_list, "task desc": task_desc_list, "test accuracy": acc_pertask, "auroc": auroc_pertask,
         "baccu": baccu_pertask})
    if log_path is not None:
        make_dir(log_path)
        logdf.to_csv(log_path)
    if save_last_model:
        make_dir(save_model_path)
        torch.save(model.state_dict(), save_model_path)


    if return_model:
        return logdf, model

    return logdf


def PUFS(centers, modelQCCPN, modelIN, threshold, device=None, max_ufea=20):
    with torch.no_grad():
        # TODO you can just concat the final feature and fix the training without using QCCPN to get the feature
        print("Searching unknown feature")
        feasize = centers.size(1)
        temp_ufea = torch.Tensor([]).to(device)
        ufea = torch.tensor([]).to(device)
        ufea90 = torch.tensor([]).to(device)
        ufea180 = torch.tensor([]).to(device)
        ufea270 = torch.tensor([]).to(device)

        nclass = centers.size(0)
        idxpermute = torch.randperm(nclass)
        ccenter = centers[idxpermute].clone()
        ccenter = (centers + ccenter)/2
        # Do genetic operation first

        temp_ufea = torch.concat((temp_ufea, ccenter), 0)
        ivc, ivc90, ivc180, ivc270 = modelIN(temp_ufea)

        output = modelQCCPN(ivc, ivc90, ivc180, ivc270)
        prob, predict = torch.max(nn.functional.softmax(output, 1), 1)
        # print(prob)
        print("threshold", threshold)
        unknown_prob = prob < threshold
        indices = torch.flatten(unknown_prob.nonzero())
        ufea = torch.concat((ufea, ivc[indices]), 0)
        ufea90 = torch.concat((ufea90, ivc90[indices]), 0)
        ufea180 = torch.concat((ufea180, ivc180[indices]), 0)
        ufea270 = torch.concat((ufea270, ivc270[indices]), 0)

        # temp_ufea = torch.Tensor([]).to(device)
        takesize = centers.size(0)
        step = 0
        while ufea.size(0) < max_ufea:
            current_size = ufea.size(0)
            if step >= 100:
                break
            print("found unknown feature size: ", ufea.size())
            idx = torch.randperm(temp_ufea.size(0))[:takesize]
            idx2 = torch.randperm(temp_ufea.size(0))[:takesize]
            candidate1 = temp_ufea[idx].clone()
            candidate2 = temp_ufea[idx2].clone()
            alpha = torch.rand(1)[0]
            candidate = alpha * candidate1 + (1-alpha) * candidate2
            temp_ufea = torch.concat((temp_ufea, candidate), 0)
            ivc, ivc90, ivc180, ivc270 = modelIN(candidate)
            output = modelQCCPN(ivc, ivc90, ivc180, ivc270)
            prob, predict = torch.max(nn.functional.softmax(output, 1), 1)
            # Less than confidence indicate the unknown
            unknown_prob = prob < threshold
            indices = torch.flatten(unknown_prob.nonzero())

            ufea = torch.concat((ufea, ivc[indices]), 0)
            ufea90 = torch.concat((ufea90, ivc90[indices]), 0)
            ufea180 = torch.concat((ufea180, ivc180[indices]), 0)
            ufea270 = torch.concat((ufea270, ivc270[indices]), 0)

            if ufea.size(0) <= current_size:
                step += 1

        return ufea, ufea90, ufea180, ufea270
def classIncrementalQCCPNTraining(device, model,  save_model_path, log_path, train_task_dl, test_task_dl, unknown_dl, feature_memory=None, load_path=None, learning_rate=0.001, start_epoch=0, n_epoch=50, batch_size=32,
                  gamma=0.1, contrastive=100, trial=5, initial_seed=1, backbone_dim=960, feature_dim=1024):

    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

    make_dir(save_model_path)
    make_dir(log_path)

    start_time = time()
    model.train()
    print("DEVICE:", device)
    model.to(device)
    n_epoch += start_epoch
    n_task = len(train_task_dl)

    # default gamma=0.1, contrastive=1000
    qccpn_loss = QCCPNLoss(gamma, contrastive, device=device)

    acc_pertask = []
    auroc_pertask = []
    baccu_pertask = []
    task_desc_list = []
    task_id_list = []
    current_seed = initial_seed
    for seed_trial in range(trial):
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)
        seed = current_seed

        inet = InverseNetworkQuadChannel(int(feature_dim/4), backbone_dim=backbone_dim)

        with torch.no_grad():
            ufea_mem = torch.Tensor([]).to(device)
            ufea90_mem = torch.Tensor([]).to(device)
            ufea180_mem = torch.Tensor([]).to(device)
            ufea270_mem = torch.Tensor([]).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for task_id in range(n_task):
            total_step = len(train_task_dl[task_id])

            if task_id > 0:
                model.extend_prototypes(n_class_per_task)
                model = model.to(device)
                ci_qccpn_loss = ClassIncrementalQCCPNLoss(gamma, contrastive, device=device)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                if feature_memory:
                    feature_memory.adjust(list(range(0, end_class)), batch_size)

            model.train()
            for epoch in range(start_epoch, n_epoch):
                total_loss = 0
                loss_step = 0
                total = 0
                correct = 0
                for i, (fea, fea90, fea180, fea270, labels, task) in enumerate(train_task_dl[task_id]):
                    fea = fea.to(device)
                    fea90 = fea90.to(device)
                    fea180 = fea180.to(device)
                    fea270 = fea270.to(device)
                    labels = labels.to(device)

                    if task_id > 0 and feature_memory:
                        # TODO check for QCCPN
                        lat_fea_mem, lat90_fea_mem, lat180_fea_mem, lat270_fea_mem, lbl_fea_mem, task_mem = feature_memory.take()
                        fea = torch.cat((fea, lat_fea_mem.to(device)), 0)
                        fea90 = torch.cat((fea90, lat90_fea_mem.to(device)), 0)
                        fea180 = torch.cat((fea180, lat180_fea_mem.to(device)), 0)
                        fea270 = torch.cat((fea270, lat270_fea_mem.to(device)), 0)
                        labels = torch.cat((labels, lbl_fea_mem.to(device)), 0)

                    output, features = model(fea, fea90, fea180, fea270, return_fea=True)

                    total += labels.size(0)
                    prob, predicted = torch.max(nn.functional.softmax(output, dim=1).data, 1)
                    correct += (predicted == labels).sum().item()

                    if task_id > 0:
                        # print("ufea_mem shape", ufea_mem.shape)
                        _, unknown_features = model(ufea_mem, ufea90_mem, ufea180_mem, ufea270_mem, return_fea=True)
                        loss = ci_qccpn_loss(features, labels, model.centers, unknown_features)
                    else:
                        loss = qccpn_loss(features, labels, model.centers)

                    optimizer.zero_grad()

                    # loss.backward(retain_graph=True)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    loss_step += 1

                print(
                    f"TASK: {task_id} - {task[0]} | Epoch [{epoch + 1}/{n_epoch}]: Loss: {total_loss / loss_step}, Accuracy: {100 * correct / total}")


            task_desc_list.append(task[0])
            task_id_list.append(task_id)

            start_class = int(task[0].split(',')[0].split(':')[1].lstrip().split("(")[-1])
            end_class = int(task[0].split(',')[1].lstrip().split(')')[0])
            n_class_per_task = end_class - start_class
            # print("n_class_pertask", n_class_per_task)
            model.to(device)
            model.eval()
            # if task_id == 0:
            test_avg_acc, auroc, baccu, threshold = evalQCCPN(device, model, task_id, test_task_dl, unknown_dl, verbose=1, return_threshold=True)
            # else:
            #     test_avg_acc, baccu = evalBACCUQCCPN(device, threshold, model, task_id, test_task_dl, unknown_dl, verbose=1)
            acc_pertask.append(test_avg_acc)
            auroc_pertask.append(auroc)
            baccu_pertask.append(baccu)
            print(f"Test Average Acc from first task 0 to current task {task_id}: {test_avg_acc}")
            print(f"AUROC from first task 0 to current task {task_id}: {auroc}")
            print(f"BACCU from first task 0 to current task {task_id}: {baccu}")
            if task_id > 0 and feature_memory:
                mse_inet = inverseNetworkTrainingQuadChannel(device, inet, model, None, None, train_task_dl[task_id], unknown_dl,
                                                         feature_memory=feature_memory, feature_mixup=False,
                                                         n_epoch=n_epoch, initial_seed=seed)
            else:
                mse_inet = inverseNetworkTrainingQuadChannel(device, inet, model, None, None, train_task_dl[task_id], unknown_dl,
                                                  feature_memory=None, feature_mixup=False,
                                                  n_epoch=n_epoch, initial_seed=seed)


            ufea_t, ufea90_t, ufea180_t, ufea270_t = PUFS(model.centers.detach(), model, inet, threshold, device=device)
            ufea_mem = torch.concat((ufea_mem, ufea_t), 0)
            ufea90_mem = torch.concat((ufea90_mem, ufea90_t), 0)
            ufea180_mem = torch.concat((ufea180_mem, ufea180_t), 0)
            ufea270_mem = torch.concat((ufea270_mem, ufea270_t), 0)

        current_seed += 1

    endtime = time()

    print(f"Total running time: {endtime - start_time}")

    logdf = pd.DataFrame(
        {"task id": task_id_list, "task desc": task_desc_list, "test accuracy": acc_pertask,  "auroc": auroc_pertask, "baccu": baccu_pertask})
    logdf.to_csv(log_path)
    return logdf