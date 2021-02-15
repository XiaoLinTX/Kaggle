'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-15 22:33:10
LastEditors: ZhangHongYu
LastEditTime: 2021-02-16 00:20:14
'''
import math
import torch
import sys
from gd_sgd import show_trace_2d
from bgd import train
from bgd import train_pytorch
from bgd import get_data
from gd_sgd import train_2d

eta = 0.4
gamma = 0.9
# RMSProp算法和AdaGrad算法的不同在于
# RMSProp算法使用了小批量随机梯度按元素平方的指数加权移动平均来调整学习率
# 若初始学习率设置为0.01，并将超参数γ设置为0.9，此时]
# st可看做是最近1/(1-0.9)=10个事件步的平方项gt⊙gt的加权平均


#  因为RMSProp算法的状态变量st是对平方项gt⊙gt
#  的指数加权移动平均，所以可以看作是最近1/(1−γ)
#  个时间步的小批量随机梯度平方项的加权平均。
#  如此一来，自变量每个元素的学习率在迭代过程中就不再一直降低（或不变）。
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1-gamma) * g1 ** 2  #  adagrad是s1 + g1 ** 2
    s2 = gamma * s2 + (1-gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2  #其余部分同adagrad


def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


def init_adagrad_states(features):
    s_w = torch.zeros((features.shape[1], 1), dtype = torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return (s_w, s_b)


def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s.data = gamma * s.data + (1-gamma) * (p.grad.data)**2
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)


if __name__ == '__main__':
    show_trace_2d(f_2d, train_2d(rmsprop_2d))
    features, labels = get_data()
    # 若初始学习率设置为0.01，并将超参数γ设置为0.9，此时]
    # st可看做是最近1/(1-0.9)=10个事件步的平方项gt⊙gt的加权平均
    # train(
    #     rmsprop,
    #     init_rmsprop_states(features),
    #     {'lr': 0.01, 'gamma':0.9},
    #     features,
    #     labels)

    # 简洁实现，通过RMSprop优化器方法，使用Pytorch提供的RMSProp算法来训练
    # 注意，超参数γ通过alpha指定
    train_pytorch(
        torch.optim.RMSprop,
        {'lr': 0.01, 'alpha': 0.9},
        features,
        labels)