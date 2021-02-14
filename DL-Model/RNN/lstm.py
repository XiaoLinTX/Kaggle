'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-14 00:21:17
LastEditors: ZhangHongYu
LastEditTime: 2021-02-14 11:52:01
'''
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as Fun
from rnn import train_and_predict_rnn
from rnn_easy import train_and_predict_rnn_pytorch, to_onehot
from sampling import load_data_jay_lyrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()


def get_params(num_inputs, num_hiddens, num_outputs):
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)
    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens, device=device, dtype=torch.float32), requires_grad=True))

    W_xi, W_hi, b_i = _three()  # 输入门参数
    W_xf, W_hf, b_f = _three()  # 遗忘门参数
    W_xo, W_ho, b_o = _three()  # 输出门参数
    W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, dtype=torch.float32), requires_grad=True)
    return nn.ParameterList([W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q])


def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), 
            torch.zeros((batch_size, num_hiddens), device=device))


class LSTM(nn.Module):
    def __init__(self, num_inputs, num_hiddens):
        super(LSTM, self).__init__()
        [self.W_xi, self.W_hi, self.b_i, self.W_xf, self.W_hf, self.b_f,
        self.W_xo, self.W_ho, self.b_o, self.W_xc, self.W_hc, self.b_c,
        self.W_hq, self.b_q ]= get_params(num_inputs, num_hiddens, num_inputs)
        for param in self.parameters():
            nn.init.normal_(param, mean=0., std=0.01)

    def forward(self, inputs, state):
        # inputs已经是小批量了，只不过是所有t的
        # inputs为(seq_length, batch_size, vocab_size)
        (H, C) = state
        outputs = []
        for X in inputs: # X 即Xt，各时间步的小批量
            I = torch.sigmoid(torch.matmul(X, self.W_xi) + torch.matmul(H, self.W_hi) + self.b_i)
            F = torch.sigmoid(torch.matmul(X, self.W_xf) + torch.matmul(H, self.W_hf) + self.b_f)
            O = torch.sigmoid(torch.matmul(X, self.W_xo) + torch.matmul(H, self.W_ho) + self.b_o)
            C_tilda = torch.tanh(torch.matmul(X, self.W_xc) + torch.matmul(H, self.W_hc) + self.b_c)
            C = F * C + I * C_tilda
            H = O * C.tanh()
            Y = torch.matmul(H, self.W_hq) + self.b_q
            outputs.append(Y)
        return outputs, (H, C)  # 除了返回输出序列，最后一个H状态外，还要返回最后一个C状态


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1) 
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state): # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        X = to_onehot(inputs, self.vocab_size) # X是个list
        Y, self.state = self.rnn(torch.stack(X), state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state


if __name__ == '__main__':
    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
    #lstm = LSTM(num_inputs, num_hiddens)
    # train_and_predict_rnn(
    #     lstm, init_lstm_state, num_hiddens,
    #     vocab_size, device, corpus_indices, idx_to_char,
    #     char_to_idx, False, num_epochs, num_steps, lr,
    #     clipping_theta, batch_size, pred_period, pred_len,
    #     prefixes)
    lr = 1e-2 # 注意调整学习率
    lstm_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
    model = RNNModel(lstm_layer, vocab_size)
    train_and_predict_rnn_pytorch(model, num_hiddens, vocab_size, device,
                                    corpus_indices, idx_to_char, char_to_idx,
                                    num_epochs, num_steps, lr, clipping_theta,
                                    batch_size, pred_period, pred_len, prefixes)
