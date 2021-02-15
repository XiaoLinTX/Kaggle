'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-15 11:19:10
LastEditors: ZhangHongYu
LastEditTime: 2021-02-15 16:30:11
'''
import torch
from gd_sgd import show_trace_2d
from bgd import get_data, train
from gd_sgd import train_2d


# eta = 0.4  # 学习率
# eta = 0.6 # 学习率调大点，自变量在竖直方向不断越过最优解并逐渐发散
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


# def gd_2d(x1, x2, s1, s2):
#     return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)


# eta , gamma = 0.4, 0.5
eta , gamma = 0.6, 0.5 #使用较大的学习率时，自变量也不再发散


def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2


features, labels = get_data()


def init_momentum_states():
    v_w = torch.zeros((features.shape[1], 1), dtype=torch.float32) 
    v_b = torch.zeros(1, dtype=torch.float32)
    return (v_w, v_b)


def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v.data = hyperparams['momentum'] * v.data + hyperparams['lr'] * p.grad.data
        p.data -= v.data

if __name__ == '__main__':
    #  show_trace_2d(f_2d, train_2d(momentum_2d))


    #  这里将momentum设为0.5，看作是特殊的小批量随机梯度下降
    #  其小批量随机梯度为最近2个时间步的2倍小批量梯度的加权平均(加权和?)
    train(
        sgd_momentum, init_momentum_states(), 
        {'lr': 0.02, 'momentum': 0.5},
        features, labels
    )

    #  momentum增大到0.9，依然可以看成是特殊的小批量随机梯度下降
    # 其小批量随机梯度为最近10个时间步的10倍小批量梯度的加权平均(加权和?)，
    # 我们先保持是学习率0.02不变
    train(
        sgd_momentum, init_momentum_states(),
        {'lr': 0.02, 'momentum': 0.9},
        features, labels
    )

    #  我们发现目标函数后期迭代过程中不够平滑，直觉上
    #  10倍小批量梯度比2倍大了5倍，我们可以试着将学习率减小到原来的1/5
    #  此时目标函数在下降了一段时间后变化更加平滑
    train(
        sgd_momentum, init_momentum_states(),
        {'lr': 0.004, 'momentum': 0.9},
        features, labels
    )
