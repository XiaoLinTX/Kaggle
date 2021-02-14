'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-02-09 16:41:11
LastEditors: ZhangHongYu
LastEditTime: 2021-02-11 01:20:55
'''
import torch
import random
import numpy as np
import zipfile
import sampling


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
    sample = corpus_indices[:20]
    # print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
    # print('indices:', sample)
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


# 随机采样 每次采样的小批量样本(一个小批量包含几段段序列)
# batch_size为每个小批量的样本数
# num_steps 为每个样本包含的时间步数
# 每次从数据里随机采样一个小批量，每次采样前都要重新初始化隐藏层
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减一是因为输出的索引x是相应输入的索引y加1
    num_examples = (len(corpus_indices)-1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始长度为num_steps的序列索引
    def _data(pos):
        return corpus_indices[pos:pos+num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i*batch_size
        batch_indices = example_indices[i: i+batch_size]
        # 相当于由每一段序列的起始索引得到每一个随机样本序列的所有索引
        # Y batch的每一个序列都是X的每一个序列往后挪一位
        # *num_steps因为序列长num_steps为了不重复
        X = [_data(j*num_steps) for j in batch_indices]
        Y = [_data(j*num_steps+1) for j in batch_indices]
        #print([_data(j*num_steps) for j in batch_indices])
        yield torch.tensor(
            X, dtype=torch.float32, device=device), torch.tensor(
                Y, dtype=torch.float32, device=device)


# 相邻采样。令相邻的两个随机小批量在原始序列上的位置相邻，
# 由此可以用一个小批量最终时间步的隐藏状态来初始化下一个
# 小批量的隐藏状态
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1 ) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i : i + num_steps]
        Y = indices[:, i + 1:i + num_steps + 1]
        yield X, Y


if __name__ == '__main__':
    my_seq = list(range(30))
    for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
        print('X: ', X, '\nY:', Y, '\n')
    for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
        print("X: ", X, '\nY:', Y, '\n')
