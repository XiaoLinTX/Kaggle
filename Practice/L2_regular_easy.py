'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-22 21:27:38
LastEditors: ZhangHongYu
LastEditTime: 2021-01-22 22:53:54
'''
import torch
import torch.nn as nn
import numpy as np
import sys
import matplotlib.pyplot as plt
# 定义一个线性回归模型


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.linear = nn.Linear(num_inputs, num_outputs)
        nn.init.normal_(self.linear.weight, mean=0, std=1)
        nn.init.normal_(self.linear.bias, mean=0, std=1)
    # forward定义前向传播

    def forward(self, X):
        return self.linear(X)


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()
# pytorch默认会对w和b都进行衰减，我们只需要对w进行衰减，故设置两个优化器


def fit_and_plot(train_iter, net, lrate, batchsize, num_epochs, loss, wd):
    train_ls, test_ls = [], []
    optimizer_w = torch.optim.SGD(
        params=[net.linear.weight], lr=lrate,  weight_decay=wd)
    optimizer_b = torch.optim.SGD(
        params=[net.linear.bias], lr=lrate,  weight_decay=0)
    for _ in range(num_epochs):
        for X, y in train_iter:
            # 添加了L2范数惩罚项
            # l求出来是一个向量，需要求和
            l = loss(net(X), y.view(-1, 1))
            # 这里原代码是mean不知为啥
            l = l.sum()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
            l.backward()
            # 对两个optimizer实例分别调用step函数，从而分别更新权重和偏差
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(X), train_labels.view(-1, 1)).mean().item())
        test_ls.append(loss(net(X), test_labels.view(-1, 1)).mean().item())
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net.linear.weight.data.norm().item())


if __name__ == '__main__':
    n_train, n_test, num_inputs = 20, 100, 200
    true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

    features = torch.randn((n_train + n_test, num_inputs))
    labels = torch.mm(features, true_w) + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                           dtype=torch.float)
    train_features, test_features = features[:n_train,
                                             :], features[n_train:, :]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    batch_size, num_epochs, lr = 1, 100, 0.003
    net = LinearNet(num_inputs, 1)
    loss = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size)

    # 观察过拟合
    fit_and_plot(train_iter, net, lr, batch_size, num_epochs, loss, 0)

    # 使用权重衰减
    fit_and_plot(train_iter, net, lr, batch_size, num_epochs, loss, 3)
