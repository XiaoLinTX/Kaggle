'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-03 12:38:34
LastEditors: ZhangHongYu
LastEditTime: 2021-02-03 13:15:28
'''
import torch
from torch import nn


def comp_con2d(conv2d, X):
    # (1, 1)代表批量大小和通道数，添加两个维度
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])


# 两侧一共填充两行或列，分别填充一行或列
if __name__ == '__main__':
    conv2d = nn.Conv2d(
        in_channels=1, out_channels = 1, kernel_size=3, padding=1)
    # conv2d = nn.Conv2d(
    #     in_channels=1, out_channels = 1, kernel_size=3, padding=2)
    # conv2d = nn.Conv2d(
    #     1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
    comp_conv2d(conv2d, X).shape
    X = torch.rand(8, 8)
    print(comp_con2d(conv2d, X).shape)