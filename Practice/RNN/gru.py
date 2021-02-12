'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-12 16:32:31
LastEditors: ZhangHongYu
LastEditTime: 2021-02-13 00:14:32
'''
import numpy as np
import torch
from rnn import train_and_predict_rnn, init_rnn_state
from torch import nn, optim
import torch.nn.functional as F
from sampling import load_data_jay_lyrics


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _one(shape):
    return nn.Parameter(torch.tensor(
        np.random.normal(0, 0.01, size=shape), device=device,
        dtype=torch.float32), requires_grad=True)


def _three(num_inputs, num_hiddens):
    return (
            _one((num_inputs, num_hiddens)),
            _one((num_hiddens, num_hiddens)),
            nn.Parameter(
                    torch.zeros(
                                num_hiddens, device=device,
                                dtype=torch.float32),
                    requires_grad=True
                )
            )


# 定义模型
class GRU(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(GRU, self).__init__()
        self.W_xz, self.W_hz, self.b_z = _three(num_inputs, num_hiddens)
        self.W_xr, self.W_hr, self.b_r = _three(num_inputs, num_hiddens)
        self.W_xh, self.W_hh, self.b_h = _three(num_inputs, num_hiddens)
        self.W_hq = _one((num_hiddens, num_outputs))
        self.b_q = torch.nn.Parameter(
            torch.zeros(num_outputs, device=device, dtype=torch.float32),
            requires_grad=True
        )
        for param in self.parameters():
            # param.requires_grad_(requires_grad=True)
            # 定义的时候就require了，这里可加可不加
            nn.init.normal_(param, mean=0, std=0.01)
            #print(param.shape)

    def forward(self, inputs, state):
        H, = state
        outputs = []
        for X in inputs:
            Z = torch.sigmoid(torch.matmul(X, self.W_xz) +
                              torch.matmul(H, self.W_hz) + self.b_z)
            R = torch.sigmoid(torch.matmul(X, self.W_xr) +
                              torch.matmul(H, self.W_hr) + self.b_r)
            H_tilda = torch.tanh(torch.matmul(X, self.W_xh) +
                                 torch.matmul(R * H, self.W_hh) + self.b_h)
            H = Z * H + (1 - Z) * H_tilda
            Y = torch.matmul(H, self.W_hq) + self.b_q
            outputs.append(Y)
        return outputs, (H,)


if __name__ == '__main__':
    num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
    (corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()
    num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
    gru = GRU(num_inputs, num_hiddens, num_outputs)
    train_and_predict_rnn(
        gru, init_rnn_state, num_hiddens,
        vocab_size, device, corpus_indices, idx_to_char,
        char_to_idx, False, num_epochs, num_steps, lr,
        clipping_theta, batch_size, pred_period, pred_len,
        prefixes)
