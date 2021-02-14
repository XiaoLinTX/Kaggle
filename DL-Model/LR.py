'''
Descripttion:LR
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-11 20:06:48
LastEditors: ZhangHongYu
LastEditTime: 2021-01-23 11:21:20
'''
import torch
import numpy as np
import random


# 定义数据读取函数
def data_iter(batchsize, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batchsize):
        j = torch.LongTensor(indices[i:min(i + batchsize, num_examples)])
        yield features[j], labels[j]


# 定义矢量计算表达式
def linreg(X, w, b):
    return torch.mm(X, w)+b


# 定义损失函数
def squared_loss(y, y_hat):
    # y为(10, ), y_hat为(10, 1)，要统一大小
    # 批量大小为10，返回 (10, 1) 张量
    return ((y.view(y_hat.size()) - y_hat)**2)/2


def sgd(params, lr, batch_size):
    for param in params:
        # 需要更新的模型参数
        param.data -= lr * param.grad / batch_size


if __name__ == '__main__':
    num_sample = 1000
    num_feature = 2

    # 训练样本
    X = torch.randn(num_sample, num_feature)
    print(X)

    # 真实模型参数
    true_w = [2, -3.4]
    true_b = 4.2

    # 训练标签，表达为w对X列向量的线性组合，加上高斯噪声
    y = true_w[0]*X[:, 0]+true_w[1]*X[:, 1]+true_b+np.random.normal(0, 0.01)

    # 初始化模型参数
    w = torch.tensor(np.random.normal(0, 0.01, (num_feature, 1)), 
    dtype=torch.float32)
    b = torch.zeros((1, 1), dtype=torch.float32)

    # 允许求梯度
    w.requires_grad_(requires_grad=True)
    b.requires_grad_(requires_grad=True)

    # 开始训练n
    lr = 0.03
    num_epoches = 3
    net = linreg
    loss = squared_loss
    batch_size = 10

    for epoch in range(num_epoches):
        for iter_X, iter_y in data_iter(batch_size, X, y):
            # 关于小批量iter_X,iter_y的总损失
            loss_v = loss(iter_y, net(iter_X, w, b)).sum()
            loss_v.backward()    # 对模型参数求梯度
            sgd([w, b], lr, batch_size)      # 一个batch一次更新，减少总的更新次数

            w.grad.data.zero_()
            b.grad.data.zero_()
        train_l = loss(net(X, w, b), y)
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
