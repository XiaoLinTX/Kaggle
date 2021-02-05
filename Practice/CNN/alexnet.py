'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-03 22:28:35
LastEditors: ZhangHongYu
LastEditTime: 2021-02-03 23:01:53
'''

import time
import torch
import torchvision.transforms as transforms
import torchvision
from torch import nn, optim
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),
            # in channels,out channels,kernel_size,stride,padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2),  # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的宽和高一致
            # 且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。
            # 除了最后的卷积层外，进一步增大了输出通道数
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        # 这里全连接层的输出个数比LeNet中的大数倍
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层，手写数字输出10类
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        # 只留下batchsize和后面压平的维度
        output = self.fc(feature.view(img.shape[0], -1))
        return output


def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


#  相比于之前从evaluate_accuracy，增加了device参数
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没有指定device就用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式，关闭dropout
                acc_sum += (
                    net(X.to(device)).argmax(dim=1) == y.to(device)
                    ).float().sum().cpu().item()
                net.train()  # 改回训练模式
            else:  # 自定义的模型，不考虑gpu，这里不会用到
                if('is_traing' in net.__code__.co_varname):
                    #  将is_training设置为False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()  # 加起来最后一起除n
            n += y.shape[0]
    return acc_sum/n


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
                   ]
    return [text_labels[int(i)] for i in labels]  # 返回5个样本的预测label


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss_v = loss(y_hat, y).sum()
            optimizer.zero_grad()
            loss_v.backward()
            optimizer.step()
            train_l_sum += loss_v.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train_acc %.3f, test_acc %.3f, time %.1f sec' %
              (epoch+1, train_l_sum/batch_count, train_acc_sum/n, test_acc, time.time() - start))


def show_fashion_mnist(images, labels):
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

def load_data_fashion_mnist(batch_size, resize=None, root='/mnt/mydisk/data'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_iter, test_iter

if __name__ == '__main__':
    # x_gpu1 = torch.rand(size=(100, 100), device='cuda:0')
    # x_gpu2 = torch.rand(size=(100, 100), device='cuda:1')
    # run(x_gpu1)
    # run(x_gpu2)
    # torch.cuda.synchronize()同步两张卡
    # 直接指定device参数可以在多卡上创建张量

    batch_size = 128
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

    net = AlexNet()
    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(
        net, train_iter, test_iter, batch_size,
        optimizer, device, num_epochs)
    print(net)