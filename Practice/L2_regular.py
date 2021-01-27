'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-22 16:31:09
LastEditors: ZhangHongYu
LastEditTime: 2021-01-27 11:14:53
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
        self.W = torch.tensor(np.random.normal(
            0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
        self.b = torch.zeros(num_outputs, dtype=torch.float)
        self.W.requires_grad_(requires_grad=True)
        self.b.requires_grad_(requires_grad=True)
        self.params = [self.W, self.b]
    # forward定义前向传播

    def forward(self, X):
        return torch.mm(X.view((-1, self.num_inputs)), self.W)+self.b
# 定义损失函数


def squared_loss(y, y_hat):
    # y为(10, ), y_hat为(10, 1)，要统一大小
    # 批量大小为10，返回 (10, 1) 张量
    return ((y.view(-1, 1) - y_hat)**2)/2


def l2_penalty(w):
    return (w**2).sum() / 2
# 算loss的时候对batchsize求和，算梯度的时候除以batchsize（为了求平均）


def sgd(params, lr, batch_size):
    for param in params:  # 需要更新的模型参数
        param.data -= lr * param.grad/batch_size


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


def fit_and_plot(lambd, train_iter, net, lr, batchsize, num_epochs):
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            # 添加了L2范数惩罚项
            # l求出来是一个向量，需要求和
            l = squared_loss(net(X), y) + lambd * l2_penalty(net.W)
            l = l.sum()

            if net.W.grad is not None:
                net.W.grad.data.zero_()
                net.b.grad.data.zero_()

            l.backward()
            sgd(net.params, lr, batch_size)
        train_ls.append(loss(net(X), train_labels).mean().item())
        test_ls.append(loss(net(X), test_labels).mean().item())
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net.W.norm().item())


if __name__ == '__main__':
    n_train, n_test, num_inputs, num_ouputs = 20, 100, 200, 1
    true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

    features = torch.randn((n_train + n_test, num_inputs))
    labels = torch.mm(features, true_w) + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                           dtype=torch.float)
    train_features, test_features = features[:n_train,
                                             :], features[n_train:, :]
    train_labels, test_labels = labels[:n_train], labels[n_train:]

    batch_size, num_epochs, lr = 1, 100, 0.003
    net, loss = LinearNet(num_inputs, num_ouputs), squared_loss

    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size)

    # 观察过拟合
    fit_and_plot(0, train_iter, net, lr, batch_size, num_epochs)

    # 使用权重衰减
    fit_and_plot(3, train_iter, net, lr, batch_size, num_epochs)
