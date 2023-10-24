from time import time

import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
import os
import sklearn
import pandas as pd
from myutils import make_dir
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import pandas as pd
import random
class LinearSoftmax(nn.Module):
    def __init__(self, n_class=75,
                 base_feature=1280,
                 input_feature=100,
                 pretrained=True):

        super(LinearSoftmax, self).__init__()
        self.num_class = n_class
        self.num_features = base_feature
        self.feature = nn.Linear(input_feature, self.num_features)
        self.relu = nn.ReLU()
        self.backbone = None
        self.final_layer = nn.Linear(self.num_features, self.num_class)

    def forward(self, x):
        x = self.relu(self.feature(x))
        x = self.final_layer(x)
        return x

    def extend_output(self, n_class):
        print("n_class", n_class)
        old_num_class = self.num_class
        self.num_class += n_class
        new_linear = nn.Linear(self.num_features, self.num_class)
        with torch.no_grad():
            new_linear.bias[:old_num_class] = self.final_layer.bias.clone()
            new_linear.weight[:old_num_class] = self.final_layer.weight.clone()

        self.final_layer = new_linear


def evalLinearSoftmax(device, model, task_id, test_task_dl, unknown_dl, verbose):
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
            for fea, labels, task in test_task_dl[t]:
                fea = fea.to(device)
                labels = labels.to(device)
                outputs = model(fea)
                label_known = torch.cat((label_known, torch.ones(labels.size(0)).to(device)), 0)
                prob, predicted = torch.max(nn.functional.softmax(outputs, dim=1).data, 1)
                prob_known = torch.cat((prob_known, prob), 0)
                predicted_known = torch.cat((predicted_known, predicted), 0)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        for fea, labels, _ in unknown_dl:
            fea = fea.to(device)
            labels = labels.to(device)
            outputs = model(fea)

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

    return 100 * correct / total, auroc, baccu


# TODO implement save and load model
def linearSoftmaxTraining(device, model,  save_model_path, log_path, train_task_dl, test_task_dl, unknown_dl, feature_memory=None, load_path=None, learning_rate=0.001, start_epoch=0, n_epoch=50, batch_size=32, trial=1, initial_seed=1):
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

    make_dir(save_model_path)
    make_dir(log_path)

    start_time = time()
    # cross_entropy_loss = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    print("DEVICE:", device)
    model.to(device)
    n_epoch += start_epoch
    n_task = len(train_task_dl)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
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
                model.extend_output(n_class_per_task)
                model.to(device)
                if feature_memory:
                    feature_memory.adjust(list(range(0, end_class)), batch_size)

            model.train()
            for epoch in range(start_epoch, n_epoch):
                total_loss = 0
                loss_step = 0
                total = 0
                correct = 0
                for i, (fea, labels, task) in enumerate(train_task_dl[task_id]):
                    # print(f"task_id = {task_id} | task = {task}") if i <= 0 else None

                    fea = fea.to(device)
                    labels = labels.to(device)

                    if task_id > 0 and feature_memory:
                        lat_fea_mem, lbl_fea_mem, task_mem = feature_memory.take()
                        # print("len lat fea mem", len(lat_fea_mem))
                        # print("lbl lat fea mem", len(lbl_fea_mem))
                        # print("lat fea mem", lat_fea_mem.shape)
                        fea = torch.cat((fea, lat_fea_mem.to(device)), 0)
                        labels = torch.cat((labels, lbl_fea_mem.to(device)), 0)

                    output = model(fea)

                    total += labels.size(0)
                    prob, predicted = torch.max(nn.functional.softmax(output, dim=1).data, 1)
                    correct += (predicted == labels).sum().item()

                    loss = cross_entropy_loss(output, labels)

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    loss_step += 1

                    # if (epoch + 1) % 10 == 0 or epoch == n_epoch-1:
                    #     print(f"Epoch [{epoch + 1}/{n_epoch}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}")

                print(
                    f"TASK: {task_id} - {task[0]} | Epoch [{epoch + 1}/{n_epoch}]: Loss: {total_loss / loss_step}, Accuracy: {100 * correct / total}")

            task_desc_list.append(task[0])
            task_id_list.append(task_id)
            start_class = int(task[0].split(',')[0].split(':')[1].lstrip().split("(")[-1])
            end_class = int(task[0].split(',')[1].lstrip().split(')')[0])
            n_class_per_task = end_class - start_class
            # print("n_class_pertask", n_class_per_task)


            model.eval()
            test_avg_acc, auroc, baccu = evalLinearSoftmax(device, model, task_id, test_task_dl, unknown_dl, verbose=1)
            acc_pertask.append(test_avg_acc)
            auroc_pertask.append(auroc)
            baccu_pertask.append(baccu)
            print(f"Test Average Acc from first task 0 to current task {task_id}: {test_avg_acc}")
            print(f"AUROC from first task 0 to current task {task_id}: {auroc}")
            print(f"BACCU from first task 0 to current task {task_id}: {baccu}")

        current_seed +=1



    endtime = time()

    print(f"Total running time: {endtime-start_time}")

    logdf = pd.DataFrame({"task id": task_id_list, "task desc": task_desc_list, "test accuracy": acc_pertask, "auroc": auroc_pertask, "baccu": baccu_pertask})
    logdf.to_csv(log_path)
    return logdf
    # return acc_pertask, auroc_pertask, baccu_pertask





