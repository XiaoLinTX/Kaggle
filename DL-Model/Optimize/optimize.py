'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-14 18:57:56
LastEditors: ZhangHongYu
LastEditTime: 2021-02-14 19:20:13
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x):
    return x * np.cos(np.pi * x )


if __name__ == '__main__':

    # 局部最小值演示
    # x = np.arange(-1.0, 2.0, 0.1)
    # fig,  = plt.plot(x, f(x))
    # fig.axes.annotate(
    #     'local minimum', xy=(-0.3, -0.25), xytext=(-0.77, -1.0),
    #     arrowprops=dict(arrowstyle='->'))
    # fig.axes.annotate(
    #     'global minimum', xy=(1.1, -0.95), xytext=(0.6, 0.8),
    #     arrowprops=dict(arrowstyle='->'))
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.show()

    # 鞍点 (saddle point)演示
    # x = np.arange(-2.0, 2.0, 0.1)
    # fig, = plt.plot(x, x**3)
    # fig.axes.annotate('saddle point', xy=(0, -0.2), xytext=(-0.52, -5.0),
    #               arrowprops=dict(arrowstyle='->'))
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    # plt.show()

    # 二维鞍点
    x, y = np.mgrid[-1: 1: 31j, -1: 1:31j]
    print(x.shape, y.shape)
    z = x**2 - y**2
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, **{'rstride': 2, 'cstride': 2})
    ax.plot([0], [0], [0], 'rx')
    ticks = [-1, 0, 1]
    plt.xticks(ticks)
    plt.yticks(ticks)
    ax.set_zticks(ticks)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()