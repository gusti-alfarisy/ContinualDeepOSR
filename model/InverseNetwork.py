from time import time

import torch
import torch.nn as nn

from myutils import make_dir
import numpy as np
import pandas as pd
import random
class InverseNetworkQuadChannel(nn.Module):
    def __init__(self, num_feature, backbone_dim=960):
        super(InverseNetworkQuadChannel, self).__init__()
        self.network = nn.Linear(num_feature, backbone_dim)
        self.network90 = nn.Linear(num_feature, backbone_dim)
        self.network180 = nn.Linear(num_feature, backbone_dim)
        self.network270 = nn.Linear(num_feature, backbone_dim)
        self.num_feature = num_feature
    def forward(self, x):
        x0 = x[:, :self.num_feature]
        x90 = x[:, self.num_feature:self.num_feature*2]
        x180 = x[:, self.num_feature*2:self.num_feature*3]
        x270 = x[:, self.num_feature*3:]
        # print("x", x.shape)
        # print("x90", x90.shape)
        # print("x180", x180.shape)
        # print("x270", x270.shape)
        return self.network(x0), self.network90(x90), self.network180(x180), self.network270(x270)


def evalINQC(device, model, modelQC, unknown_dl, verbose):
    mse_tensor = torch.Tensor([]).to(device)
    with torch.no_grad():
        for fea, fea90, fea180, fea270, labels, task in unknown_dl:
            fea = fea.to(device)
            fea90 = fea90.to(device)
            fea180 = fea180.to(device)
            fea270 = fea270.to(device)
            # labels = labels.to(device)
            _, feature = modelQC(fea, fea90, fea180, fea270, return_fea=True)
            out, out90, out180, out270 = model(feature)

            mse = nn.functional.mse_loss(torch.concat((out, out90, out180, out270), 1), torch.concat((fea, fea90, fea180, fea270), 1))
            # print("mse", mse)
            mse_tensor = torch.cat((mse_tensor, torch.unsqueeze(mse, 0)), 0)
            # print("mse tensor", mse_tensor)
    # print("mse tensor", mse_tensor)
    # print("mse tensor mean", torch.mean(mse_tensor))
    return torch.mean(mse_tensor).item()

def inverseNetworkTrainingQuadChannel(device, model, modelQC, save_model_path, log_path, train_dl, unknown_dl, feature_memory=None, load_path=None, learning_rate=0.001, start_epoch=0, n_epoch=50, initial_seed=1, feature_mixup=False, alpha=0.5):
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

    if save_model_path:
        make_dir(save_model_path)

    if log_path:
        make_dir(log_path)

    start_time = time()
    # cross_entropy_loss = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    print("DEVICE:", device)
    model.to(device)
    n_epoch += start_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    current_seed = initial_seed
    torch.manual_seed(current_seed)
    np.random.seed(current_seed)
    random.seed(current_seed)
    model.train()

    mseLoss = nn.MSELoss()
    modelQC.eval()
    for epoch in range(start_epoch, n_epoch):
        total_loss = 0
        loss_step = 0
        total = 0
        for i, (fea, fea90, fea180, fea270, labels, task) in enumerate(train_dl):
            # print(f"task_id = {task_id} | task = {task}") if i <= 0 else None

            fea = fea.to(device)
            fea90 = fea90.to(device)
            fea180 = fea180.to(device)
            fea270 = fea270.to(device)
            labels = labels.to(device)
            sizefea = fea.size(1)
            batchsize = fea.size(0)
            if feature_mixup:
                cf = torch.concat((fea, fea90, fea180, fea270), 1)
                cf2 = cf.clone()
                randidx = torch.randperm(batchsize)
                cf2 = cf2[randidx]
                mu_fea_cf = cf * alpha + cf2 * (1 - alpha)
                mu_fea = mu_fea_cf[:, :sizefea]
                mu_fea90 = mu_fea_cf[:, sizefea:sizefea*2]
                mu_fea180 = mu_fea_cf[:, sizefea*2:sizefea*3]
                mu_fea270 = mu_fea_cf[:, sizefea*3:]

                fea = torch.concat((fea, mu_fea), 0)
                fea90 = torch.concat((fea90, mu_fea90), 0)
                fea180 = torch.concat((fea180, mu_fea180), 0)
                fea270 = torch.concat((fea270, mu_fea270), 0)

            if feature_memory:
                lat_fea_mem, lat90_fea_mem, lat180_fea_mem, lat270_fea_mem, lbl_fea_mem, task_mem = feature_memory.take()
                fea = torch.cat((fea, lat_fea_mem.to(device)), 0)
                fea90 = torch.cat((fea90, lat90_fea_mem.to(device)), 0)
                fea180 = torch.cat((fea180, lat180_fea_mem.to(device)), 0)
                fea270 = torch.cat((fea270, lat270_fea_mem.to(device)), 0)
                labels = torch.cat((labels, lbl_fea_mem.to(device)), 0)

            _, feature = modelQC(fea, fea90, fea180, fea270, return_fea=True)

            out, out90, out180, out270 = model(feature)

            total += labels.size(0)
            loss = mseLoss(torch.concat((out, out90, out180, out270), 1), torch.concat((fea, fea90, fea180, fea270), 1))

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loss_step += 1

            # if (epoch + 1) % 10 == 0 or epoch == n_epoch-1:
            #     print(f"Epoch [{epoch + 1}/{n_epoch}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}")

        print(
            f"Epoch [{epoch + 1}/{n_epoch}]: Loss: {total_loss / loss_step}")

    model.eval()
    mse = evalINQC(device, model, modelQC, unknown_dl, verbose=1)
    print("MSE Eval INQC:", mse)
    endtime = time()
    print("total runnning time inverse network:", endtime-start_time)

    return mse