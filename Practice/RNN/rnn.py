'''
Descripttion: rnn的从零实现
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-10 11:22:20
LastEditors: ZhangHongYu
LastEditTime: 2021-02-12 23:46:34
'''

import time
import math
import numpy as np
import torch
import zipfile
import sampling
from torch import nn, optim
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Rnn(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(Rnn, self).__init__()
        self.W_xh = torch.ones((num_inputs, num_hiddens), device=device)   
        self.W_hh = torch.ones((num_hiddens, num_hiddens), device=device)
        self.W_hq = torch.ones((num_hiddens, num_outputs), device=device)
        self.b_h = torch.zeros(num_hiddens, device=device)
        self.b_q = torch.zeros(num_outputs, device=device)
        for param in self.parameters():
            param.requires_grad_(requires_grad=True)
            nn.init.normal_(param, mean=0, std=0.01)

    def forward(self, inputs, state):
        # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
        outputs = []
        # Rnn的inputs的一个batch样本是一个序列
        # batch大小为2就是两个seq
        # inputs是一个batch，
        # 那么就是(seq_length, 2, vocab_size)
        # (seq_lenth, batch_size, vocab_size)
        # 能保障时序的意思
        H,  = state
        for X in inputs:
            H = torch.tanh(torch.matmul(X, self.W_xh)+torch.matmul(H, self.W_hh)+self.b_h)
            # 隐藏层的计算公式，除了考虑常规的Xt*Wxh
            # 还要考虑到时序上Ht-1到Ht-1的映射
            Y = torch.matmul(H, self.W_hq) + self.b_q
            # 输出层的公式仍属于常规
            outputs.append(Y)
        return outputs, (H,)


def load_data_jay_lyrics():
    with zipfile.ZipFile('/mnt/mydisk/LocalCode/data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    # print(corpus_chars[:40])
    # print(type(corpus_chars[:40]))

    # 把换行符替换成空格，使用前一万个字符来训练
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    # print(corpus_chars[0:10000])

    # 建立字符索引, 构建词典,vocab_size为词典中不同字符个数
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)

    # 将训练集中字符转换成索引
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    # sample = corpus_indices[:20]
    # print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
    # print('indices:', sample)
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


# 词典对应的索引->one-hot向量，向量长度为词典大小vocab_size
def one_hot(x, n_class, dtype=torch.float32):
    # X shape:(batch), output_shape:(batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


def to_onehot(X, n_class):
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


# 初始化隐藏层状态，注意是(batch_size, num_hiddens)
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device, requires_grad=True), )


def predict_rnn(
                prefix, num_chars, rnn,
                init_rnn_state, num_hiddens,
                vocab_size, device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出做为当前时间步
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            # 如果后面还接着有字符，则添加后面的字符
            output.append(char_to_idx[prefix[t+1]])
        else:
            # 否则，就用预测的字符替代之
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


# 裁剪梯度，把模型参数梯度的元素拼成一个向量g
# 设裁剪的阈值为theta
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    #print(norm.shape)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm .sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta/norm)


def sgd(params, lr, batch_size):
    for param in params:  # 需要更新的模型参数
        param.data -= lr * param.grad/batch_size


# 困惑度:对交叉损失函数做指数运算
# 我们的模型训练函数使用困惑度评价模型
def train_and_predict_rnn(
    rnn, init_rnn_state, num_hiddens,
    vocab_size, device, corpus_indices, idx_to_char,
    char_to_idx, is_random_iter, num_epochs, num_steps,
    lr, clipping_theta, batch_size, pred_period, pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = sampling.data_iter_random
    else:
        data_iter_fn = sampling.data_iter_consecutive
    
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:
            # 如使用相邻采样，有必要在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:
                # 采用随机采样，每个小样本之前要初始化隐层状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
            # 否则就要使用detach从计算图分离隐藏状态，这是为了
            # 使模型参数的梯度计算只依赖一次迭代读取的小批量
                for s in state:
                    s.detach_()
            inputs = to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size,vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state)
            # 拼接之后形状为(num_steps*batch_size, vocab_size)
            # 将num_steps那一维度都连在一起
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            # 使用交叉熵损失计算平均分类误差
            loss_v = loss(outputs, y.long())
            # print(loss_v)
            # print(y.shape[0])
            # # 梯度清0
            # if rnn.parameters()[0].grad is not None:
            for param in rnn.parameters():
                print(param)
                param.grad.data.zero_()

            loss_v.backward()
            grad_clipping(rnn.parameters(), clipping_theta, device)  # 裁剪梯度
            sgd(rnn.parameters(), lr, loss_v)  #　因为误差已经取过均值，梯度不用再做平均
            l_sum += loss_v.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print(
                'epoch %d, perplexity %f, item %.2f sec' % (epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                                        prefix, pred_len, rnn,
                                        init_rnn_state, num_hiddens,
                                        vocab_size, device, idx_to_char, char_to_idx))



if __name__ == '__main__':
    # indices也是下标，在数学金融领域常用
    (corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()
    # x = torch.tensor([0, 2])
    # print(one_hot(x, vocab_size))
    X = torch.arange(10).view(2, 5).to(device)  # 一个 batchsize 输入的X
    inputs = to_onehot(X, vocab_size)
    # print(len(inputs), inputs[0].shape)

    # 时序问题的输入输出维度都是vocab_size
    rnn = Rnn(vocab_size, 256, vocab_size)

    num_hiddens = 256
    # 初始的隐藏层状态
    state = init_rnn_state(inputs[0].shape[0], num_hiddens, device)

    # # 我们观察第一个时间步输入的一个batchsize
    # outputs, state_new = rnn(inputs, state)


    # out = predict_rnn(
    #             '分开', 10, rnn, init_rnn_state,
    #             num_hiddens, vocab_size, device,
    #             idx_to_char, char_to_idx
    # )    
    # print(out)

    # grad_cipping(rnn.parameters(), 0.1, device)
    
    # 每经过50个迭代周期便根据当前训练的模型创作一段长为50的歌词
    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

    train_and_predict_rnn(
        rnn, init_rnn_state, num_hiddens,
        vocab_size, device, corpus_indices, idx_to_char,
        char_to_idx, True, num_epochs, num_steps, lr,
        clipping_theta, batch_size, pred_period, pred_len, prefixes
    )
