'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-03 17:25:24
LastEditors: ZhangHongYu
LastEditTime: 2021-02-03 19:39:01
'''
import time
import torch
from torch import nn, optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 在卷积层块中输入的高和宽在逐层减小。
# 卷积层由于使用高和宽均为5的卷积核，
# 从而将高和宽分别减小4，而池化层则将高和宽减半，
# 但通道数则从1增加到16。全连接层则逐层减少输出个数，
# 直到变成图像的类别数1
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
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


def train(net, train_iter, test_iter, loss, num_epochs, batch_size, lr=None, optimizer=None):
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


if __name__ == '__main__':
    mnist_train = torchvision.datasets.FashionMNIST(root='/mnt/mydisk/data'', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='/mnt/mydisk/data'', train=False, download=True, transform=transforms.ToTensor())

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






    net = LeNet()
    print(net)