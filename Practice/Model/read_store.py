'''
Descripttion: Tensor和模型的读写
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-04 20:42:30
LastEditors: ZhangHongYu
LastEditTime: 2021-02-04 21:46:04
'''


import torch
from torch import nn


# Tensor的读写
# save使用pickle将序列化的对象保存到disk
# 包括模型，张量，字典等
# load使用pickle unpicke工具将pickle的对象文件反序列
# 化为内存


# 读写模型
# Module的可以学习参数（权重和偏置）包含在参数中
# 可通过 model.parameters() 访问
# state_dict是一个从参数名称映射到参数Tensor的字典对象
# 是个OrderedDict（其实Py3 Dict都是Ordered，此处为了保证）
# 比如对于以下网络
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


if __name__ == '__main__':

    # 保存Tensor
    x = torch.ones(3)
    torch.save(x, 'Practice/Model/x.pt')
    x2 = torch.load('Practice/Model/x.pt')
    print(x2)

    # 保存Tensor组成的列表
    y = torch.zeros(4)
    torch.save([x, y], 'Practice/Model/xy.pt')
    xy_list = torch.load('Practice/Model/xy.pt')
    print(xy_list)

    # 存储并读取一个从字符串映射到Tensor的字典
    torch.save({'x': x, 'y': y}, 'Practice/Model/xy_dict.pt')
    xy = torch.load('Practice/Model/xy_dict.pt')
    print(xy)

    #单独保存一个字符对象也是可以的
    torch.save('hello', 'Practice/Model/str.pt')
    xy = torch.load('Practice/Model/str.pt')
    print(xy)

    mlp = MLP()
    print(mlp.state_dict())

    #只有具有可学习参数的层(卷积层、线性层等)
    # 才有state_dict中的条目。
    # 优化器(optim)也有一个state_dict，
    # 其中包含关于优化器状态以及所使用的超参数的信息。
    optimizer = torch.optim.SGD(
        mlp.parameters(),
        lr=0.001,
        momentum=0.9    
    )
    print(optimizer.state_dict())
    
    #torch的模型保存与加载
    # 推荐：仅保存和加载模型参数state_dict
    MODEL_PATH = 'Practice/Model/model_dict.pt'
    torch.save(mlp.state_dict(), MODEL_PATH)
    mlp = MLP()
    mlp.load_state_dict(torch.load(MODEL_PATH))

    # 不推荐：保存和加载整个模型
    torch.save(mlp, MODEL_PATH)
    model = torch.load(MODEL_PATH)
