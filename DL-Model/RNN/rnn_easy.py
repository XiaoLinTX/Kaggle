'''
Descripttion: rnn的简易实现
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-11 10:18:06
LastEditors: ZhangHongYu
LastEditTime: 2021-02-12 11:51:23
'''
import time
import math
import zipfile
import numpy as np
import torch
import sampling
from rnn import one_hot, grad_clipping
from torch import nn, optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Rnn(nn.Module):
    def __init__(self, num_inputs, num_hiddens):
        # num_inputs必然等于num_outputs，不必赘述
        # 与上一节不同，这里的rnn直接所有num_steps的输入
        # 输入形状为(num_steps, batch_size, vocab_size)
        # rnn_layer会返回输出和隐藏状态h
        # 1.注意，这个输出为各个时间步输出的隐藏状态，(num_steps, batch_size, num_hiddens)
        # 并不返回输出层的计算
        # 2.而我们说的前向计算返回的隐藏状态指的是隐藏层在
        # 最后一个num_step的隐藏状态，记忆了每一层hidden_layer信息
        # LSTM的隐藏状态是(h, c)，即hidden_state和cell_state
        super(Rnn, self).__init__()
        self.rnn_layer = nn.RNN(
            input_size = num_inputs,
            hidden_size = num_hiddens
        )
        self.bidirectional = 0

    def forward(self, X, state):
        return self.rnn_layer(X, state)


def to_onehot(X, n_class):
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


#　我们接下来尝试实现完整的LSTM实例,has-a嵌套类关系
class LSTM(nn.Module):
    def __init__(self, vocab_size):
        super(LSTM, self).__init__()
        self.rnn = Rnn(vocab_size, 256)
        self.hidden_size = 256*(2 if self.rnn.bidirectional else 1)
        self.vocab_size = vocab_size
        # 加一个全连接层涉及输出的计算
        self.dense = nn.Linear(self.hidden_size, self.vocab_size)
        self.state = None

    def forward(self, inputs, state):  # inputs:(batch, seq_len)
        # 获取 one-hot向量表示
        # 被编码为长seq_len的列表，列表元素为tensor(batch_size, vocab_size)
        X = to_onehot(inputs, self.vocab_size)
        # torch.stack(X)为tensor(seq_len, batch_size, vocab_size)
        Y, self.state = self.rnn(torch.stack(X), state)
        # 全连接层会首先将Y的形状变成(num_steps*batch_size, num_hiddens)
        # 它的输出形状为(num_steps*batch_size, vocab_size)
        # 因为三维张量没法矩阵乘啊
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state


# 预测模型，区别在于前向计算和初始化隐藏状态的函数接口
def predict_rnn_pytorch(
                        prefix, num_chars, model, vocab_size, device,
                        idx_to_char, char_to_idx):
    state = None
    output = [ char_to_idx[prefix[0]] ]  # output 会记录prefix加上输出
    for t in range(num_chars + len(prefix)-1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        # 输入(1, 1)
        if state is not None:
            if isinstance(state, tuple):  #  LSTM, state:(h,c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)

        (Y, state) = model(X, state)
        # 输出 (1, vocab_size)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t+1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


# 实现训练函数，算法同上一节的一样，但这里只使用了相邻采样来读取数据
def train_and_predict_rnn_pytorch(
    model, num_hiddens, vocab_size, device,
    corpus_indices, idx_to_char, char_to_idx,
    num_epochs, num_steps, lr, clipping_theta,     
    batch_size, pred_period, pred_len, prefixes
):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = sampling.data_iter_consecutive(
            corpus_indices, batch_size, num_steps, device
        )   # 相邻采样
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离出隐藏状态，这是为了
                # 使模型的参数梯度计算只依赖一次迭代读取的小批量序列(防止梯度开过大)
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()
            (output, state) = model(X, state)
            # output:形状为(numsteps*batch_size, vocab_size)

            # Y的形状是(batch_size,seq_len)，转置后再变成长度为:
            # seq_len * batch_size的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)

            loss_v = loss(output, y.long())

            optimizer.zero_grad()
            loss_v.backward()
            # 梯度裁剪
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += loss_v.item() * y.shape[0]
            n += y.shape[0]

        try:
            perplexity = math.exp(l_sum/n)
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print(' epoch %d, perplexity %f, time %.2f sec' % (
                    epoch + 1, perplexity, time.time() - start
                )
            )
            for prefix in prefixes:
                print('-', predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device,
                    idx_to_char, char_to_idx
                ))


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


if __name__ =='__main__':
    (corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()
    num_steps = 35
    batch_size = 2
    state = None
    # X = torch.rand(num_steps, batch_size, vocab_size)
    # rnn = Rnn(vocab_size, 256)
    # , state_new = rnn(X, state)
    # print(Y.shape, len(state_new), state_new[0].shape)

    model = LSTM(vocab_size).to(device)
    # predict_rnn_pytorch('分开', 10, model, vocab_size, device, idx_to_char, char_to_idx)
    num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2 # 注意这里的学习率设置
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    train_and_predict_rnn_pytorch(
        model, 256, vocab_size, device,
        corpus_indices, idx_to_char, char_to_idx,
        num_epochs, num_steps, lr, clipping_theta,
        batch_size, pred_period, pred_len, prefixes)




