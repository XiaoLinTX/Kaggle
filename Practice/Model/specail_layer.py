'''
Descripttion: 自定义神经网络中的层
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-04 16:43:39
LastEditors: ZhangHongYu
LastEditTime: 2021-02-04 20:39:45
'''

import torch
from torch import nn


# 可以通过Module类自定义神经网络中的层，从而可以被重复调用。
# 不含模型参数的自定义层
class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


# 我们也可以用自定义层来构造更复杂的模型
class ModelComplex(nn.Module):
    def __init__(self, **kwargs):
        super(ModelComplex, self).__init__(**kwargs)
        self.func = nn.Sequential(
            nn.Linear(8, 128),
            CenteredLayer()
        )

    def forward(self, x):
        y = self.func(x)
        return y


# 含模型参数的自定义曾
# 我们要将参数定义为Parameter使它被自动添加到模型的参数列表里
# 除了Parameter，还可以使用ParameterList和ParameterDict
# 分别定义参数的列表和字典
class MyListDense(nn.Module):
    def __init__(self):
        super(MyListDense, self).__init__()
        self.params = nn.ParameterList(
            [nn.Parameter(torch.randn(4, 4)) for i in range(3)]
        )
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x


# 而ParameterDict接收一个Parameter实例的字典作为输入
# 然后得到一个参数字典，即可按字典规则使用，update()新增参数，
# keys()返回所有键值，items()返回所有键值对
class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
                'linear1': nn.Parameter(torch.randn(4, 4)),
                'linear2': nn.Parameter(torch.randn(4, 1))
            }
        )
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))})  # 新增

    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])


# 我们也可以使用自定义层构造模型。它和PyTorch的其他层在使用上很类似
class ModelComplex2(nn.Module):
    def __init__(self, **kwargs):
        super(ModelComplex2, self).__init__(**kwargs)
        self.func = nn.Sequential(
            MyDictDense(),
            MyListDense()
        )

    def forward(self, x):
        y = self.func(x)
        return y


if __name__ == '__main__':
    layer = CenteredLayer()
    res = layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))

    m_complex = ModelComplex()
    y = m_complex(torch.rand(4, 8))
    res = y.mean().item()

    net = MyListDense()
    print(net)

    net2 = MyDictDense()
    print(net2)
    # 根据传入的键值来进行不同的前向传播
    x = torch.ones(1, 4)
    print(net2(x, 'linear1'))
    print(net2(x, 'linear2'))
    print(net2(x, 'linear3'))

    net3 = ModelComplex2()
    print(net3)
    print(net3(x))