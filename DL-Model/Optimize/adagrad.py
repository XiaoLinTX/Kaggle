'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-15 22:33:10
LastEditors: ZhangHongYu
LastEditTime: 2021-02-16 00:12:25
'''
import math
import torch
import sys
from gd_sgd import show_trace_2d
from bgd import train
from bgd import train_pytorch
from bgd import get_data
from gd_sgd import train_2d

# eta = 0.4
# eta 改为2，自变量更为讯速地逼近了最优解
eta = 2
def adagrad_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    #  前两项是自变量梯度，共同组成梯度向量g

    s1 += g1 **2
    s2 += g2 **2  #  梯度向量的平方按元素累加到s向量上  

    #   最终结果累加到自变量向量x上
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def init_adagrad_states(features):
    s_w = torch.zeros((features.shape[1], 1), dtype = torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return (s_w, s_b)


def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        #  遍历每一个param，即遍历参数向量的分量
        s.data += (p.grad.data**2)
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)

if __name__ == '__main__':
    show_trace_2d(f_2d, train_2d(adagrad_2d))
    features, labels = get_data()
    train(
        adagrad,
        init_adagrad_states(features),
        {'lr': 0.1},
        features,
        labels)

    # 简洁实现，通过Adagrad优化器方法，使用Pytorch提供的AdaGrad算法来训练
    train_pytorch(torch.optim.Adagrad, {'lr': 0.1}, features, labels)