'''
Descripttion: Dropout的easy实现
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-23 16:42:17
LastEditors: ZhangHongYu
LastEditTime: 2021-01-23 22:28:42
'''
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import sys


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hiddens1)
        self.linear2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.linear3 = nn.Linear(num_hiddens2, num_outputs)
        for param in self.parameters():
            nn.init.normal_(param, mean=0, std=0.01)
        self.drop_prob1, self.drop_prob2 = 0.2, 0.5
        self.dropout1 = nn.Dropout(self.drop_prob1)
        self.dropout2 = nn.Dropout(self.drop_prob2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        H1 = self.relu(self.linear1(X.view(X.shape[0], -1)))
        H1 = self.dropout1(H1)
        H2 = self.relu(self.linear2(H1))
        H2 = self.dropout2(H2)
        Out = self.linear3(H2)
        return self.softmax(Out)


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        if isinstance(net, torch.nn.Module):
            # 我们这个满足isinstance条件，无需用到is_training参数
            net.eval()  # 评估模式, 这会关闭dropout
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train()  # 改回训练模式
        else:
            # 自定义的模型
            # 如果有is_training这个参数
            if('is_training' in net.__code__.co_varnames):
                # 将is_training设置成False
                acc_sum += (net(X, is_training=False).argmax(
                    dim=1) == y).float().sum().item()
            else:
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def train(
        net, train_iter, test_iter, loss,
        num_epochs, batch_size, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            loss_v = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif net.parameters() is not None:
                for param in net.parameters():
                    param.grad.data.zero_()

            loss_v.backward()
            if optimizer is None:
                torch.optim.SGD(net.parameters(), lr)
            else:
                optimizer.step()

            train_l_sum += loss_v.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train_acc %.3f, test_acc %.3f' % (
            epoch+1, train_l_sum/n, train_acc_sum/n, test_acc))


def show_fashion_mnist(images, labels):
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    mnist_train = torchvision.datasets.FashionMNIST(
        root='/mnt/mydisk/data', train=True,
        download=True,
        transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(
        root='/mnt/mydisk/data', train=False, download=True,
        transform=transforms.ToTensor())

    # print(type(mnist_train))
    # print(len(mnist_train),len(mnist_test))
    # feature, label = mnist_train[0]
    # print(feature.shape, label)

    batch_size = 256

    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4  # 多线程读取数据

    train_iter = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size,
        shuffle=True, num_workers=num_workers)
    # X = torch.arange(16).view(2, 8)
    # res1 = dropout(X, 0)
    # res2 = dropout(X, 0.5)
    # res3 = dropout(X, 1.0)
    num_inputs = 784
    num_outputs = 10
    num_hiddens1 = 256
    num_hiddens2 = 256
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    # 这里使用torch自带的sgd，学习率可照常设置
    num_epochs, lr, batch_size = 5, 0.5, 256
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr)

    train(
        net, train_iter, test_iter, loss,
        num_epochs, batch_size, lr, optimizer)
