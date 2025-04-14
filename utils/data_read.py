import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_config(yml_path):

    with open(yml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_data(config, set_name='train'):
    data_name = config['dataset']
    if __name__ == "utils.data_read":
        if set_name == 'train':
            data_pth = './data/processed/' + data_name + '/X_train.csv'
            Y_pth = './data/processed/' + data_name + '/Y_train.csv'
        elif set_name == 'test':
            data_pth = './data/processed/' + data_name + '/X_test.csv'
            Y_pth = './data/processed/' + data_name + '/Y_test.csv'
    else:
        if set_name == 'train':
            data_pth = '../data/processed/' + data_name + '/X_train.csv'
            Y_pth = '../data/processed/' + data_name + '/Y_train.csv'
        elif set_name == 'test':
            data_pth = '../data/processed/' + data_name + '/X_test.csv'
            Y_pth = '../data/processed/' + data_name + '/Y_test.csv'

    df_X = pd.read_csv(data_pth)
    df_Y = pd.read_csv(Y_pth)
    Y = df_Y.values.squeeze()

    if not np.issubdtype(Y.dtype, np.integer):
        raise ValueError("标签必须是整数类型。")

    min_Y = Y.min()
    if min_Y != 0:
        Y = Y - min_Y
    num_classes = Y.max() + 1

    if Y.min() < 0 or Y.max() >= num_classes:
        raise ValueError(f"标签值必须在 [0, {num_classes - 1}] 范围内。当前标签范围: [{Y.min()}, {Y.max()}]")

    num_continuous = config['num_continuous_features']
    # num_discrete = config['num_discrete_features']

    X_cont = df_X.iloc[:, 0:num_continuous]
    X_disc = df_X.iloc[:, num_continuous:]

    X = np.hstack([X_cont, X_disc])
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.long)

    return X_tensor, Y_tensor


if __name__ == "__main__":
    config = load_config('../configs/swissmetro.yml')
    X, Y = load_data(config, set_name='test')
    print(f"特征矩阵X的形状: {X.shape}")  # 应为 (N, D_total)
    print(f"标签向量Y的形状: {Y.shape}")  # 应为 (N,)