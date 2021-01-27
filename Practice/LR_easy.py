'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-11 20:06:48
LastEditors: ZhangHongYu
LastEditTime: 2021-01-27 11:14:38
'''
import torch
from time import time
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import random
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
#定义一个线性回归模型


class LinearNet(nn.Module):
    def __init__(self, num_feature):
        super().__init__()
        self.linear = nn.Linear(num_feature, 1)  # 定义模型的一层
    #forward定义前向传播

    def forward(self, x):
        y = self.linear(x)
        return y


if __name__ == '__main__':
    num_sample = 1000
    num_feature = 2

    #训练样本
    X = torch.randn(num_sample, num_feature)

    #真实模型参数
    true_w = [2, -3.4]
    true_b = 4.2

    #训练标签，表达为w对X列向量的线性组合，加上高斯噪声
    y = true_w[0]*X[:, 0]+true_w[1]*X[:, 1]+true_b+np.random.normal(0, 0.01)

    #开始训练n
    lr = 0.03
    num_epoches = 3
    net = LinearNet(num_feature)
    loss = nn.MSELoss()
    batch_size = 10

    dataset = Data.TensorDataset(X, y)
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)  # 随机读取小批量

    #初始化模型参数
    init.normal_(net.linear.weight, mean=0, std=0.01)  # 标准正态分布
    init.constant_(net.linear.bias, val=0)

    #定义模型优化器
    optimizer = optim.SGD(net.parameters(), lr=0.03)

    for epoch in range(num_epoches):
        for iter_X, iter_y in data_iter:
            l = loss(iter_y.view(-1, 1), net(iter_X))  # 关于小批量iter_X,iter_y的总损失
            optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
            l.backward()  # 对模型参数求梯度
            optimizer.step()
        print('epoch %d, loss %f' % (epoch + 1, l.item()))
