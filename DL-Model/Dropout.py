import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import sys


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens1 = num_hiddens1
        self.num_hiddens2 = num_hiddens2
        self.W1 = torch.tensor(np.random.normal(0, 0.01, (
            num_inputs, num_hiddens1)), dtype=torch.float)
        self.b1 = torch.zeros(num_hiddens1, dtype=torch.float)
        self.W2 = torch.tensor(np.random.normal(0, 0.01, (
            num_hiddens1, num_hiddens2)), dtype=torch.float)
        self.b2 = torch.zeros(num_hiddens2, dtype=torch.float)
        self.W3 = torch.tensor(np.random.normal(0, 0.01, (
            num_hiddens2, num_outputs)), dtype=torch.float)
        self.b3 = torch.zeros(num_outputs, dtype=torch.float)
        self.W1.requires_grad_(requires_grad=True)
        self.b1.requires_grad_(requires_grad=True)
        self.W2.requires_grad_(requires_grad=True)
        self.b2.requires_grad_(requires_grad=True)
        self.W3.requires_grad_(requires_grad=True)
        self.b3.requires_grad_(requires_grad=True)
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self.drop_prob1, self.drop_prob2 = 0.2, 0.5

    # 此时X行n为样本个数，列m为输出的预测标签个数
    def relu(self, X):
        # 两个tensor取最大，运用广播机制
        return torch.max(input=X, other=torch.tensor(0.0))

    def softmax(self, X):
        # 先对矩阵所有元素取exp
        X_exp = X.exp()
        partition = X_exp.sum(dim=1, keepdim=True)
        # 计算矩阵每行的和，保留列，得到n*1矩阵
        # 广播机制，n*m可相除以n*1
        return X_exp / partition

    def dropout(self, X, drop_prob):
        X = X.float()
        assert 0 <= drop_prob <= 1
        keep_prob = 1 - drop_prob
        # 这种情况把元素全都丢弃
        if keep_prob == 0:
            return torch.zeros_like(X)
        # 随机算法技巧，将概率经过大小比较转换为示性变量I
        mask = (torch.rand(X.shape) < keep_prob).float()
        return mask * X / keep_prob

    def forward(self, X, is_training=True):
        H1 = self.relu(torch.mm(X.view((
            -1, self.num_inputs)), self.W1) + self.b1)
        if is_training:
            H1 = self.dropout(H1, self.drop_prob1)
        H2 = self.relu(torch.mm(H1.view((
            -1, self.num_hiddens1)), self.W2) + self.b2)
        if is_training:
            H2 = self.dropout(H2, self.drop_prob2)
        Out = torch.mm(H2, self.W3) + self.b3
        return self.softmax(Out)


def sgd(params, lr, batch_size):
    for param in params:  # 需要更新的模型参数
        param.data -= lr * param.grad / batch_size


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
            elif net.params is not None and net.params[0].grad is not None:
                for param in net.params:
                    param.grad.data.zero_()
            loss_v.backward()
            if optimizer is None:
                sgd(net.params, lr, batch_size)
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
        root='/mnt/mydisk/data'', train=True,
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
    # 学习率设计成一百，抵消sgd中的/batchsize
    num_epochs, lr, batch_size = 5, 100, 256
    loss = torch.nn.CrossEntropyLoss()

    train(
        net, train_iter, test_iter, loss,
        num_epochs, batch_size, lr)
