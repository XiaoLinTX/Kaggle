'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-03 23:48:46
LastEditors: ZhangHongYu
LastEditTime: 2021-02-03 23:49:36
'''
import torch

X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
H, W_hh = torch.randn(3, 4), torch.randn(4, 4)
r1 = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
r2 = torch.matmul(torch.cat((X, H), dim=1), torch.cat((W_xh, W_hh), dim=0))
print(r1)
print(r2)
