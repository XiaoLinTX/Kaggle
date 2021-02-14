'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-04 11:09:53
LastEditors: ZhangHongYu
LastEditTime: 2021-02-04 16:42:28
'''
import torch
from torch import nn
from torch.nn import init


# 继承Module类
# 重载了Module类的__init__函数和forward函数
# Module的子类既可以是一个Layer，也可以是一个Model
class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(4, 3) # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(3, 1)  # 输出层
        #默认权重随机初始化

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


# 我们用Sequential辅助构造模型
class MLP_Seq(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP_Seq, self).__init__(**kwargs)
        self.func = nn.Sequential(
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
        )

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        return self.func(x)


# 以上另外返回的param的类型为torch.nn.parameter.Parameter
# 其实这是Tensor的子类，和Tensor不同的是如果一个Tensor是Parameter
# 那么它会自动被添加到模型的参数列表里
# 否则需要self.params =[self.W,self.b]来手动添加
class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(20, 20))
        # 在参数列表
        self.weight2 = torch.rand(20, 20)  # 不在参数列表

    def forward(self, x):
        pass


# PyTorch内置初始化方法,inplace且不记录梯度
# def normal_(tensor, mean=0, std=1):
#     with torch.no_grad():
#         return tensor.normal_(mean, std)
# 自定义inplace初始化方法,我们令权重有一半概率初始化为0，
# 有另一半概率初始化为[−10,−5]和[5,10]两个区间里
# 均匀分布的随机数。
def my_init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()


# 共享模型参数：Module类的forward函数里多次调用同一个层。
# 此外，如果我们传入Sequential的模块是同一个Module实例的话参数也是共享的
class MLP_Shared(nn.Module):
    def __init__(self, **kwargs):
        super(MLP_Shared, self).__init__(**kwargs)
        linear = nn.Linear(1, 1, bias=False)
        self.func = nn.Sequential(
            linear,
            linear
        )

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        return self.func(x)



if __name__ =='__main__':
    net = MLP()  # pytorch已进行默认初始化
    net2 = MLP_Seq()
    net3 = MyModel()
    print(net)
    # 打印MLP_Seq会自动加上序号索引
    print(net2)
    X = torch.rand(2, 4)
    Y = net2(X).sum()
    print(Y)

    # 访问模型参数
    print(type(net.parameters()))  # 生成器
    for param in net.parameters():
        print(param.size())

    # 访问模型参数
    print(type(net2.parameters()))  # 生成器
    for param in net2.parameters():
        print(param.size())

    print(type(net.named_parameters()))
    for name, param in net.named_parameters():
        print(name, param.size())

    print(type(net2.named_parameters()))
    for name, param in net2.named_parameters():
        print(name, param.size())

    # # 对于MLP_Seq访问net中和单层的参数
    # 因为这里是单层的所以没有了层数索引的前缀
    # for name, param in net2[0].named_parameters():
    #     print(name, param.size(), type(param))

    n = MyModel()
    for name, param in n.named_parameters():
        print(name)

    # Parameter是Tensor
    # 可以根据data来访问参数数值，用grad来访问参数梯度。
    # weight_0 = list(net2[0].parameters())[0]
    # print(weight_0.data)
    # print(weight_0.grad)  # 反向传播前梯度为None
    # Y.backward()
    # print(weight_0.grad)

    # 参数初始化
    for name, param in net.named_parameters():
        if 'weight' in name:
            init.normal_(param, mean=0, std=0.01)
            print(name, param.data)
    # 用常数来初始化
    for name, param in net.named_parameters():
        if 'bias' in name:
            init.constant_(param, val=0)
            print(name, param.data)
    # 两个bias分别为tensor([0.,0.,0.])和tensor([0.])

    # 检验自定义初始化方法
    for name, param in net.named_parameters():
        if 'weight' in name:
            my_init_weight_(param)
            print(name, param.data)

    # 我们还可以通过改变这些参数的data来改写模型参数值同时不会影响梯度:
    for name, param in net.named_parameters():
        if 'bias' in name:
            param.data += 1
            print(name, param.data)

    # 共享模型参数测试
    net4 = MLP_Shared()
    print(net4)
    for name, param in net4.named_parameters():
        init.constant_(param, val=3)
        print(name, param.data)
    # 内存中，两个线性层其实是一个对象
    # print(id(net4[0]) == id(net4[1]))
    # print(id(net4[0].weight) == id(net4[1].weight))

    # 因为模型参数里包含了梯度，所以在反向传播计算时，
    # 这些共享的参数的梯度是累加的
    # x = torch.ones(1, 1)
    # y = net4(x).sum()
    # print(y)
    # y.backward()
    # print(net4[0].weight.grad) # 单次梯度是3，两次所以就是6

    #题外话：PyTorch定义模型时必须指定输入形状