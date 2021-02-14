'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-03 17:04:45
LastEditors: ZhangHongYu
LastEditTime: 2021-02-03 17:23:53
'''

import torch
from torch import nn


# 在输出大小上和卷积运算类似
def pool2d(X, pool_size, mode='max'):
    X = X.float()
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0]-p_h+1, X.shape[1]-p_w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i+p_h, j:j+p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i+p_h, j:j+p_w].mean()
    return Y

if __name__ == '__main__':
    X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    print(pool2d(X, (2, 2)))

    # 下面演示池化层填充和步幅
    X = torch.arange(16, dtype=torch.float).view((1, 1, 4, 4))
    
    # 默认步幅和池化窗口形状相同
    pool2d = nn.MaxPool2d(3)
    print(pool2d(X))

    # 改变步幅
    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    print(pool2d(X))

    # 非正方形池化窗口
    pool2d = nn.MaxPool2d((2, 4), padding=(1, 2), stride=(2, 3))
    pool2d(X)


    #处理多通道输入数据时，池化层对每个输入通道分别池化，而不是像卷积层那样将
    #将各通道的输入按通道相加
    X = torch.cat((X, X+1), dim=1)
    #池化后，我们发现输出通道数仍然是2。
    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    print(pool2d(X))