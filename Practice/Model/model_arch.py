'''
Descripttion: MLP
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-03 23:57:15
LastEditors: ZhangHongYu
LastEditTime: 2021-02-04 20:31:35
'''

import torch
from torch import nn
from collections import OrderedDict #py3字典都是有序的，此处针对py2

# 继承Module类
# 重载了Module类的__init__函数和forward函数
# Module的子类既可以是一个Layer，也可以是一个Model
class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256) # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层
        #默认权重随机初始化

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


# Sequential继承自Module类，可辅助构建模型
#可以接收一个子模块的有序字典（OrderedDict）或者一系列子模块作为参数
class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self, input):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成员
        for module in self._modules.values():
            input = module(input)
        return input


# 我们用Sequential辅助构造模型
class MLP_Seq(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP_Seq, self).__init__(**kwargs)
        self.func = MySequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        return self.func(x)


# 用Module_list初始化吗模型，相比seq在前向传播时无序
# 加入Module list的参数会被自动添加到网络中，
# 单个Linear的parameters()为torch.size([10])
# 此处的Module_list的为torch.size([10,10])
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x


# ModuleDict接收一个子模块的字典作为输入, 
# 然后也可以类似字典那样进行添加访问操作，前向也无序，需自定义forward函数
# 其参数也会被自动添加到整个网络
class MLP_Seq(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP_Seq, self).__init__(**kwargs)
        self.func = nn.ModuleDict({
            'linear': nn.Linear(784, 256),
            'act': nn.ReLU(),
        })
    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        return self.func(x)


# 复杂模型直接继承Module类更灵活
class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        #常数权重，不可训练的模型参数
        self.rand_weight = torch.rand((20, 20), requires_grad=False) # 不可训练参数（常数参数）
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # 使用创建的常数参数，以及nn.functional中的relu函数和mm函数
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

        # 共享模型参数，复用全连接层。等价于两个全连接层共享参数
        x = self.linear(x)
        # 控制流，这里我们需要调用item函数来返回标量进行比较
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()


# 因为FancyMLP和Sequential类都是Module类的子类，所以我们可以嵌套调用它们。
class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU()) 

    def forward(self, x):
        return self.net(x)


class MLP_nest(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP_nest, self).__init__(**kwargs)
        self.func = nn.Sequential(
            NestMLP(),
            nn.Linear(30, 20),
            FancyMLP())

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        return self.func(x)



if __name__ == '__main__':
    X = torch.rand(2, 784)
    net1 = MLP()
    net2 = MLP_Seq()
    # print(net1)
    # print(net2)
    # print(net1(X))
    # print(net2(X))

    # X = torch.rand(2, 20)
    # net = FancyMLP()
    # print(net)
    # print(net(X))

    net = MLP_nest()
    X = torch.rand(2, 40)
    print(net)
    print(net(X))