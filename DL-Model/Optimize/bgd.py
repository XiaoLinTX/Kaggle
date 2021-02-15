'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-14 22:50:08
LastEditors: ZhangHongYu
LastEditTime: 2021-02-15 11:18:07
'''
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from torch import nn, optim

def get_data():
    data = np.genfromtxt('/mnt/mydisk/LocalCode/data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
    torch.tensor(data[:1500, -1], dtype=torch.float32)  
    # 前1500个样本(每个样本5个特征)

# 状态输入:states，超参数:hyperparams
# 和前面训练函数对损失求sum，这里除batch_size不同的是，这次
# 在训练函数里对各小批量样本的损失求平均了，这里不需要除

def sgd(params, states, hyperparams):
    for p in params:
        p.data -= hyperparams['lr'] * p.grad.data


def squared_loss(y, y_hat):
    # y为(10, ), y_hat为(10, 1)，要统一大小
    # 批量大小为10，返回 (10, 1) 张量
    return ((y.view(-1, 1) - y_hat)**2)/2


def linreg(X, w, b):
    return torch.mm(X, w)+b


def train(optimizer_fn, states, hyperparams, features, labels,
              batch_size=10, num_epochs=2):
    # 初始化模型
    net, loss = linreg, squared_loss

    w = torch.nn.Parameter(torch.tensor(np.random.normal(0, 0.01, size=(features.shape[1], 1)), dtype=torch.float32),
                           requires_grad=True)
    b = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)

    def eval_loss():
        return loss(net(features, w, b), labels).mean().item()

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            l = loss(net(X, w, b), y).mean()  # 使用平均损失

            # 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()

            l.backward()
            optimizer_fn([w, b], states, hyperparams)  # 迭代模型参数
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())  # 每100个样本记录下当前训练误差
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


#  简洁实现，用optimizer实例调用优化算法，以下我们通过optimizer_fn
#  和超参数optimizer_hyperparams来创建optimizer实例
#  本函数与原书不同的是这里第一个参数优化器函数而不是优化器的名字
#  例如: optimizer_fn=torch.optim.SGD, optimizer_hyperparams={"lr": 0.05}
def train_pytorch(optimizer_fn, optimizer_hyperparams, features, labels,
                    batch_size=10, num_epochs=2):
    # 初始化模型
    net = nn.Sequential(
        nn.Linear(features.shape[-1], 1)
    )
    loss = nn.MSELoss()
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)

    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            # 除以2是为了和train保持一致, 因为squared_loss中除了2
            # 注意optim.SGD里已经平均了，但没除以2
            l = loss(net(X).view(-1), y) / 2 

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def train_sgd(features, labels, lr, batch_size, num_epochs=2):
    #train(sgd, None, {'lr': lr}, features, labels, batch_size, num_epochs)
    train_pytorch(optim.SGD, {'lr': lr}, features, labels, batch_size, num_epochs)

if __name__ == '__main__':
    features, labels = get_data()
    # print(features.shape)
    train_sgd(features, labels, 1, 1500, 6)
    train_sgd(features, labels, 0.005, 1, 6)
    train_sgd(features, labels, 0.05, 10, 6)