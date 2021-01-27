'''
Descripttion: Process data
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-23 21:50:58
LastEditors: ZhangHongYu
LastEditTime: 2021-01-27 11:13:45
'''
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hiddens)
        self.linear2 = nn.Linear(num_hiddens, num_outputs)
        self.relu = nn.ReLU()
        for param in self.parameters():
            nn.init.normal_(param, mean=0, std=0.01)

    def forward(self, X):
        H = self.relu(self.linear1(X))
        Out = self.linear2(H)
        return Out


def preprocess(all_features):
    # 提取出对连续数值的特征，画重点!
    numeric_features = all_features.dtypes[all_features.dtypes != object].index
    # 对数值特征的每一列x进行标准化处理
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x-x.mean())/(x.std())
    )
    # 标准化后数值特征均值为0，用0来替换缺失值
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    # 将离散特征转为指示特征
    all_features = pd.get_dummies(all_features, dummy_na=True)
    return all_features


# 比赛评价模型的对数均方根误差
def log_rmse(net, features, labels, loss):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


# 模型训练
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size, loss):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # 这里使用了Adam优化算法
    optimizer = torch.optim.Adam(
        params=net.parameters(), lr=learning_rate,
        weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            loss_v = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            loss_v.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels, loss))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels, loss))
    return train_ls, test_ls


# k折交叉验证
def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        # 如果(j+1)*fold_size超出了shape[0]，slice会自动舍弃
        idx = slice(j * fold_size, (j+1)*fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid


def k_fold(
        k, X_train, y_train, num_epochs,
        learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0

    # k次k折交叉验证，每折验证时都要重新初始化模型
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        num_outputs = 1
        num_hiddens = 256
        net = LinearNet(X_train.shape[1], num_outputs, num_hiddens)
        loss = nn.MSELoss()
        train_ls, valid_ls = train(
            net, *data, num_epochs,
            learning_rate, weight_decay, batch_size, loss)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            semilogy(
                range(1, num_epochs+1), train_ls, 'epochs',
                'rmse', range(1, num_epochs+1), valid_ls, ['train', 'valid'])
        print("fold %d, train rmse %f, valid rmse %f" % (
            i, train_ls[-1], valid_ls[-1]))
    return train_l_sum/k, valid_l_sum/k


def train_and_pred(
                train_features, test_features, train_labels, test_data,
                num_epoches, learning_rate, weight_decay, batch_size):
    num_outputs = 1
    num_hiddens = 256
    net = LinearNet(train_features.shape[1], num_outputs, num_hiddens)
    loss = nn.MSELoss()
    train_ls, _ = train(
        net, train_features, train_labels,
        None, None, num_epoches, lr, weight_decay, batch_size, loss
    )
    semilogy(range(1, num_epoches+1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    joblib.dump(net, '/mnt/mydisk/LocalCode/model/PredictPrices')
    net = joblib.load('/mnt/mydisk/LocalCode/model/PredictPrices')
    pred = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(pred.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']])
    submission.to_csv('Predict-Prices/data/submission.csv', index=False)


if __name__ == '__main__':
    train_data = pd.read_csv('Predict-Prices/data/train.csv')
    test_data = pd.read_csv('Predict-Prices/data/test.csv')

    # 合并训练和测试特征，除去train的房价和id号，除去test的id号，train和test都取79个特征
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    all_features = preprocess(all_features)

    # 再次拆分训练和测试特征并生成训练和测试特征张量，并生成训练标签张量
    n_train = train_data.shape[0]
    train_features = torch.tensor(
        all_features[:n_train].values, dtype=torch.float
        )
    test_features = torch.tensor(
        all_features[n_train:].values, dtype=torch.float
        )
    train_labels = torch.tensor(
        train_data.iloc[:, -1].values, dtype=torch.float
    ).view(-1, 1)

    # k折交叉验证选择模型并调好超参数
    k, num_epochs, lr, weight_decayay, batch_size = 5, 100, 5, 0, 64
    train_l, valid_l = k_fold(
        k, train_features, train_labels,
        num_epochs, lr, weight_decayay, batch_size)
    print("%d-fold validation:avg train rmse %f, avg valid rmse %f" % (
        k, train_l, valid_l))

    # 以上只是用，下面才是真的训练并预测
    train_and_pred(
        train_features, test_features, train_labels, test_data,
        num_epochs, lr, weight_decayay, batch_size)
