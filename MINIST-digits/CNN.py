import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

EPOCH = 10
BATCH_SIZE = 1
LR = 0.001  # 学习率


def readdata():
    data = pd.read_csv(
        '/media/lonelyprince7/mydisk/CV-dataset/MINIST/train.csv')
    rows = data.shape[0]
    cols = data.shape[1]
    train = data.iloc[0:rows*2//3, :]
    test = data.iloc[rows*2//3:rows, :]

    train_label = train.iloc[:, 0].to_numpy()
    train_x = train.iloc[:, 1:].to_numpy()
    train_x_ = []
    for x in train_x:
        train_x_.append(x.reshape(28, 28).tolist())
    train_x_ = np.array(train_x_)

    test_label = test.iloc[:, 0].to_numpy()
    test_x = test.iloc[:, 1:].to_numpy()
    test_x_ = []
    for x in test_x:
        test_x_.append(x.reshape(28, 28).tolist())
    test_x_ = np.array(test_x_)
    # 画一个图片
    # plt.imshow(train_x_[0], cmap='gray')
    # plt.title('%i' % train_label[0])
    # plt.show()
    return train_x_, train_label, test_x_, test_label


if __name__ == '__main__':
    # 训练集(28000,28,28),测试集(14000, 28, 28)
    train_x, train_label, test_x, test_label = readdata()
    train_dataset = Data.TensorDataset(
        torch.tensor(train_x), torch.tensor(train_label))
    test_dataset = Data.TensorDataset(
        torch.tensor(test_x), torch.tensor(test_label))

    # #输入训练集(28000,28,28),经dataloader操作后,每次迭代得到的是(5, 28, 28),表示本次取出5个样本，每个样本(28,28),当然，标签随之打乱
    train_loader = Data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 测试集(14000, 28, 28)->(14000, 1, 28, 28),
    test_x = Variable(torch.unsqueeze(torch.tensor(test_x), dim=1)).type(
        torch.cuda.FloatTensor)/255.
    test_label = torch.tensor(test_label).cuda()

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(         # 输入张量(1, 28, 28)
                nn.Conv2d(
                    in_channels=1,              # 输入通道
                    out_channels=16,            # 卷积核数量，亦输出通道个数
                    kernel_size=5,
                    stride=1,
                    # 若想卷积后图像大小不变, padding=(kernel_size-1)/2 若stride=1
                    padding=2,
                ),                              # 输出张量 (16, 28, 28)
                nn.ReLU(),                      # 激活函数
                # 池化，选取2x2区域内最大值,输出张量(16, 14, 14)
                nn.MaxPool2d(kernel_size=2),
            )
            self.conv2 = nn.Sequential(  # 输入张量(16, 14, 14)
                nn.Conv2d(16, 32, 5, 1, 2),  # 输出张量(32, 14, 14)
                nn.ReLU(),  # 激活函数
                nn.MaxPool2d(2),  # 输出张量(32, 7, 7)
            )
            self.out = nn.Linear(32 * 7 * 7, 10)  # 全连接网络层，输出10个类别

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            # 将conv2的输出压扁成(batch_size, 32 * 7 * 7)
            x = x.view(x.size(0), -1)
            output = self.out(x)
            return output, x  # 返回(batch_size,32*7*&)可视化

    cnn = CNN()
    cnn.cuda()
    print(cnn)  # 网络结构

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # 优化所有cnn 参数
    loss_func = nn.CrossEntropyLoss()                       # 所有目标label没有独热编码

    from matplotlib import cm
    try:
        from sklearn.manifold import TSNE
        HAS_SK = True
    except:
        HAS_SK = False
        print('Please install sklearn for layer visualization')

    def plot_with_labels(lowDWeights, labels):
        plt.cla()
        X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
        for x, y, s in zip(X, Y, labels):
            c = cm.rainbow(int(255 * s / 9))
            plt.text(x, y, s, backgroundcolor=c, fontsize=9)
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.min(), Y.max())
        plt.title('Visualize last layer')
        plt.show()
        plt.pause(0.01)

    plt.ion()

    # 训练模型并测试准确率
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):  # normalize x prin
            x = Variable(x).cuda()
            b_x = Variable(torch.unsqueeze(x, dim=1)).type(
                torch.cuda.FloatTensor)/255.  # (5,28,28)->(5,1,28,28)
            b_y = Variable(y).cuda()   # (1)标签

            output = cnn(b_x)[0]            # cnn 输出,直接输出标签?
            loss = loss_func(output, b_y)   # 交叉熵
            optimizer.zero_grad()           # 为这步清空梯度
            loss.backward()                 # 反向传播，计算梯度
            optimizer.step()                # 应用梯度

            if step % 50 == 0:
                test_output, last_layer = cnn(test_x)  # last layer是啥子?
                pred_y = torch.max(test_output, 1)[1].data.squeeze()
                sum_ = 0.
                for (i, j) in zip(pred_y, test_label):
                    if i == j:
                        sum_ = sum_ + 1
                accuracy = sum_ / float(test_label.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' %
                      loss.item(), '| test accuracy: %.2f' % accuracy)
                if HAS_SK:
                    # Visualization of trained flatten layer (T-SNE)
                    tsne = TSNE(perplexity=30, n_components=2,
                                init='pca', n_iter=5000)
                    plot_only = 500
                    low_dim_embs = tsne.fit_transform(
                        last_layer.cpu().data.numpy()[:plot_only, :])
                    labels = test_label.cpu().numpy()[:plot_only]
                    plot_with_labels(low_dim_embs, labels)
    plt.ioff()

    # print 10 predictions from test data
    test_output, _ = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].cpu().data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')
