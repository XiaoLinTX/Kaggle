'''
Descripttion: GPU使用
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-04 21:47:40
LastEditors: ZhangHongYu
LastEditTime: 2021-02-04 23:45:47
'''
import torch
from torch import nn


if __name__ == '__main__':
    # 查看GPU是否可用
    print(torch.cuda.is_available())

    # 查看GPU数量
    print(torch.cuda.device_count())

    # 查看GPU索引号，从0开始
    print(torch.cuda.current_device())

    # 根据索引号查看GPU名字
    print(torch.cuda.get_device_name(0))

    x = torch.tensor([1, 2, 3])
    x = x.cuda(0)  # CPU复制到GPU，cuda(i)表示第i块GPU
    print(x)
    print(x.device)  # 可以查看设备索引号

    # 当然，我们可以直接在创建的时候指定设备
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'cpu'
    )
    x = torch.tensor([1, 2, 3], device=device)
    # or
    x = torch.tensor([1, 2, 3]).to(device)
    print(device)
    print(x)

    # 对GPU上的数据做运算，结果还是存放在GPU
    y = x**2
    print(y)

    # 而存在GPU的数据不能直接和CPU的数据进行计算
    # z = y + x.cpu()


    # 模型也可以转换到GPU上
    net = nn.Linear(3, 1)
    # 可以看到第一个权重存在CPU上
    print(list(net.parameters())[0].device)

    # 将其转换到GPU上,此处为inplace
    net.cuda()
    print(list(net.parameters())[0].device)

    # 同样的，需要模型输入的Tensor和模型都在同一设备上
    #，否则会报错
    x = torch.rand(2, 3).cuda()
    print(net(x))

