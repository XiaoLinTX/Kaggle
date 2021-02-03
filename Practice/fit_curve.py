'''
Descripttion: 采用神经网络拟合收入曲线
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-29 18:33:41
LastEditors: ZhangHongYu
LastEditTime: 2021-01-30 01:05:14
'''
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit
import numpy as np


model_save_root = '/mnt/mydisk/LocalCode/model/MCM20/'
batch_size = 1
num_workers = 4
num_epochs = 100
lr = 0.01


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.linear1 = nn.Linear(num_inputs, num_hiddens)
        self.linear2 = nn.Linear(num_hiddens, num_outputs)
        self.relu = nn.ReLU()
        for param in self.parameters():
            nn.init.normal_(param, mean=0, std=0.01)

    # 此时X行n为样本个数，列m为输出的预测标签个数
    def forward(self, X):
        H = self.relu(self.linear1(X.view(-1, self.num_inputs)))
        # 这里再确保输入的是二维张量
        Out = self.linear2(H)
        return Out


def train(net, train_iter, loss, num_epochs, batch_size, lr, optimizer, label):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            loss_v = loss(y_hat, y).sum()
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif net.parameters() is not None:
                for param in net.parameters():
                    param.grad.data.zero_()
            loss_v.backward()
            if optimizer is None:
                torch.optim.SGD(net.parameters(), lr)
            else:
                optimizer.step()
    torch.save(net.state_dict(), model_save_root+label+'_fish.pt')


if __name__ == '__main__':
    stomoway_data = pd.read_csv('MCM20/data/stomoway_fish.csv')
    shetland_data = pd.read_csv('MCM20/data/shetland_fish.csv')
    x = stomoway_data['year'].to_numpy().reshape(-1, 1)
    y1 = stomoway_data['fish'].to_numpy().reshape(-1, 1)
    y2 = shetland_data['fish'].to_numpy().reshape(-1, 1)

    plt.scatter(x, y1, color='red') 
    # plt针对表示为(..., 1)的矩阵形式向量时，会自动帮你去掉dim=1
    plt.scatter(x, y2, color='blue')

    #  神经网络的输入特征和标签都要经过标准化和反标准化
    mm1, mm2, mm3 = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
    x = mm1.fit_transform(x)
    y1 = mm2.fit_transform(y1)
    y2 = mm3.fit_transform(y2)

    dataset1 = torch.utils.data.TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y1).float())   # 直接Tensor(x, dtype=...)是针对list

    dataset2 = torch.utils.data.TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y2).float())

    train_iter1 = torch.utils.data.DataLoader(
        dataset1, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)
    train_iter2 = torch.utils.data.DataLoader(
        dataset2, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)

    net1 = LinearNet(num_inputs=1, num_outputs=1, num_hiddens=256)
    net2 = LinearNet(num_inputs=1, num_outputs=1, num_hiddens=256)
    optimizer1 = torch.optim.SGD(net1.parameters(), lr)
    optimizer2 = torch.optim.SGD(net2.parameters(), lr)
    loss = nn.MSELoss()
    train(
            net1, train_iter1, loss, num_epochs, batch_size, lr, optimizer1,
            'stomoway')
    train(
            net2, train_iter2, loss, num_epochs, batch_size, lr, optimizer2,
            'shetland')

    m_state_dict1 = torch.load(model_save_root+'stomoway_fish.pt')
    net1.load_state_dict(m_state_dict1)
    m_state_dict2 = torch.load(model_save_root+'shetland_fish.pt')
    net2.load_state_dict(m_state_dict2)

    y_pred1 = net1(torch.from_numpy(x).float())
    y_pred2 = net2(torch.from_numpy(x).float())

    x = mm1.inverse_transform(x)
    y_pred1 = mm2.inverse_transform(y_pred1.data.numpy())
    y_pred2 = mm3.inverse_transform(y_pred2.data.numpy())

    plt.plot(
        x,
        y_pred1,
        color='red', linestyle='--', lw=1)
    plt.plot(
        x,
        y_pred2,
        color='blue', linestyle='--', lw=1)
    plt.savefig('MCM20/data/fish.png')