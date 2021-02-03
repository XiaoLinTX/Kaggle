'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-03 13:26:59
LastEditors: ZhangHongYu
LastEditTime: 2021-02-03 16:51:48
'''
import torch
import torch.nn as nn


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0]-h+1, X.shape[1]-w+1), dtype=float)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


# 多输入通道
def corr2d_multi_in(X, K):
    # 沿着X和K的第0维（通道维）分别计算再相加
    res = corr2d(X[0, :, :], K[0, :, :])  # 用通道0先初始化res
    for i in range(1, X.shape[0]):
        res += corr2d(X[i, :, :], K[i, :, :])
    return res


def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历， 每次同输入X做互相关运算，
    # 所有结果使用stack函数合并在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K])


# 1*1卷积层相当于全连接层
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0] # 输出的通道数等于卷积核的通道数
    # 把输入X的h,w压成一维  
    X = X.view(c_i, h*w)
    K = K.view(c_o, c_i) #左乘
    Y = torch.mm(K, X) 
    return Y.view(c_o, h, w)


if __name__ == '__main__':
    # X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    #           [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    # K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

    # # print(corr2d_multi_in(X, K))

    # # K+1是K所有元素加一，链接成一个3通道的卷积核
    # # 如果只有多输入单输出，一个卷积核分别和各通道卷积相加，输入一个2d_map
    # # 如果要保留三通道，则多个卷积核分别和各通道卷积，执行单通道操作×3，输入3×2d_map
    # K = torch.stack([K, K+1, K+2])
    # print(corr2d_multi_in_out(X, K))
    

    # 以下是1*!卷积层内容
    X = torch.rand(3, 3, 3)
    K = torch.rand(2, 3, 1, 1)
    Y1 = corr2d_multi_in_out_1x1(X, K)
    Y2 = corr2d_multi_in_out(X, K)

    print((Y1 - Y2).norm().item() < 1e-6)
    
