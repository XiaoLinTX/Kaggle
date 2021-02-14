'''
Descripttion: CNN
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-03 11:38:49
LastEditors: ZhangHongYu
LastEditTime: 2021-02-03 12:17:10
'''

import torch
from torch import nn


class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias










def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0]-h+1, X.shape[1]-w+1), dtype=float)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


if __name__ == '__main__':
    X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    K = torch.tensor([[0, 1], [2, 3]])
    print(corr2d(X, K))