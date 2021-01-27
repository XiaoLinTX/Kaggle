'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-11 20:01:10
LastEditors: ZhangHongYu
LastEditTime: 2021-01-27 11:16:11
'''
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import os
import argparse
import json


def read_data():
    file_path = '/media/lonelyprince7/mydisk/NLP-dataset/task3/ELE/train/train0001.json'
    with open(file_path) as f:
        js = json.load(f)  # js是转换后的字典
        sentences = js['article'].split('.')
        answers = js['answers']
        options = js['options']
    res = []
    for anslist, ans in zip(options, answers):
        if ans == 'A':
            res.append(anslist[0].lower())
        if ans == 'B':
            res.append(anslist[1].lower())
        if ans == 'C':
            res.append(anslist[2].lower())
        if ans == 'D':
            res.append(anslist[3].lower())
    return sentences, res


PAD, MASK, CLS, SEP = '[PAD]', '[MASK]', '[CLS]', '[SEP]'


def to_bert_input(tokens, bert_tokenizer):
    token_idx = torch.tensor(bert_tokenizer.convert_tokens_to_ids(tokens))
    sep_idx = tokens.index('[SEP]')
    segment_idx = token_idx * 0
    segment_idx[(sep_idx + 1):] = 1
    mask = (token_idx != 0)
    return token_idx.unsqueeze(0), segment_idx.unsqueeze(0), mask.unsqueeze(0)


parser = argparse.ArgumentParser()
parser.add_argument('--topk', type=int, default=1,
                    help='show top k predictions')
if __name__ == '__main__':
    args = parser.parse_args()
    bert_tokenizer = BertTokenizer(
        vocab_file='/media/lonelyprince7/mydisk/NLP-dataset/bert_models/bert-base-uncased-vocab.txt')
    bert_model = BertForMaskedLM.from_pretrained(
        '/media/lonelyprince7/mydisk/NLP-dataset/bert_models/bert-base-uncased.tar.gz')
    sentences, res = read_data()
    print(res)
    predict_res = []
    mask_cnt = 0
    for sentence in sentences:
        sentence = sentence.strip()
        sentence = sentence.replace('_', '[MASK]')
        # print(sentence)
        tokens = bert_tokenizer.tokenize(sentence)
        if len(tokens) == 0:
            continue
        if tokens[0] != CLS:
            tokens = [CLS] + tokens
        if tokens[-1] != SEP:
            tokens.append(SEP)
        token_idx, segment_idx, mask = to_bert_input(tokens, bert_tokenizer)
        with torch.no_grad():
            logits = bert_model(token_idx, segment_idx,
                                mask, masked_lm_labels=None)
        logits = logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        for idx, token in enumerate(tokens):
            if token == MASK:
                mask_cnt += 1
                print('Top {} predictions for {}th {}:'.format(
                    args.topk, mask_cnt, MASK))
                topk_prob, topk_indices = torch.topk(probs[idx, :], args.topk)
                topk_tokens = bert_tokenizer.convert_ids_to_tokens(
                    topk_indices.cpu().numpy())
                for prob, tok in zip(topk_prob, topk_tokens):
                    print('{} {}'.format(tok, prob))
                    predict_res.append(tok)
                print('='*80)
    cnt = correct_cnt = 0
    for item1, item2 in zip(res, predict_res):
        if item1 == item2:
            correct_cnt = correct_cnt+1
            print(item1+'对了!')
        cnt = cnt+1
    print('correct rate is:%.2f' % (correct_cnt/cnt))
