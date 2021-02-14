'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-21 23:01:10
LastEditors: ZhangHongYu
LastEditTime: 2021-01-27 10:26:15
'''
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


def fit_and_plot(train_features, test_features, train_labels, test_labels):
    num_epochs, loss = 100, torch.nn.MSELoss()
    #net可接受任意的样本数，但是对输入特征有要求
    net = torch.nn.Linear(train_features.shape[-1], 1)
    #由Linear文档可知，pytorch已经将参数初始化了

    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
    print("final epoch:train_loss:train_loss",
          train_ls[-1], 'test_loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data,
          '\nbias:', net.bias.data)


if __name__ == '__main__':
    #三阶多项式拟合
    n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
    features = torch.randn((n_train s + n_test, 1))
    #类似于多项式线性空间，多项式每一阶的项即对应一个维度
    poly_features = torch.cat((features, torch.pow(features, 2),
                               torch.pow(features, 3)), 1)
    labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
              + true_w[2] * poly_features[:, 2])  # 可以理解成线性组合，也可以理解成w向量与每一个样本做内积
    # print(features[:2])
    # print(poly_features[:2])
    # print(labels[:2])

    #正常拟合
    fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
                 labels[:n_train], labels[n_train:])

    #欠拟合，只有一个特征的线性多项式
    fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train],
                 labels[n_train:])

    #训练样本不足使其过拟合，这里只使用两个样本来训练，使得模型过于复杂
    fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2],
                 labels[n_train:])
