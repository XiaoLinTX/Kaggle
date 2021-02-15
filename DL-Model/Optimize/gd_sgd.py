'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-14 20:14:37
LastEditors: ZhangHongYu
LastEditTime: 2021-02-14 22:46:22
'''
import numpy as np
import torch
import matplotlib.pyplot as plt
import math

# 使用x=10做为初始值，设ita=0.2。梯度下降迭代10次
def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        x -= eta * (2 * x)
        results.append(x)
    print('epoch 10, x:', x)
    return results


def show_traces(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    plt.plot(f_line, [x * x for x in f_line])
    plt.plot(res, [x * x for x in res], '-o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# 多维梯度下降
# 对于f(x) = x1**2 + 2 * x2**2
# 从[-5, -2]开始对x迭代20次
def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i+1, x1, x2))
    return results


def show_trace_2d(f, results):
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))    
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

eta = 0.1
def f_2d(x1, x2):
    return x1 ** 2 + 2 * x2 **2


def gd_2d(x1, x2, s1, s2):
    #  这里对求偏导数硬编码
    return (x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0)


# 我们在梯度中添加均值为0的随机噪声来模拟随机梯度下降
# 在实际中，这些噪声通常指训练数据集中无意义的干扰
def sgd_2d(x1, x2, s1, s2):
   return (x1 - eta * (2 * x1 + np.random.normal(0.1)),
            x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0) 



if __name__ == '__main__':
    # 可以尝试更改学习率查看搜索变化
    #  show_traces(gd(1.1))

    # # 2d情况下
    # show_trace_2d(f_2d, train_2d(gd_2d))

    #随机梯度下降
    show_trace_2d(f_2d, train_2d(sgd_2d))