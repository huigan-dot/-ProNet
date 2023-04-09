#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
from torch.utils.data import Dataset
import h5py
import random
import pandas as pd


def load_h5_2D(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        print(h5_path)
        head = list(hf.keys())
        X = np.transpose(np.array(hf.get(head[0]), dtype=np.float32))
        Y = np.transpose(np.array(hf.get(head[1]), dtype=np.float32))
        if X.ndim == 3:
            X1 = X.swapaxes(1, 2)  # 将数组n个维度中两个维度进行调换
            X1 = X1[:, :, np.newaxis, :]
            Y1 = Y.astype(np.float32)
        elif Y.ndim == 3:
            X1 = Y.swapaxes(1, 2)  # 将数组n个维度中两个维度进行调换
            X1 = X1[:, :, np.newaxis, :]
            Y1 = X.astype(np.float32)
        else:
            raise RuntimeError("维度错误")
        index = [i for i in range(len(Y1))]
        random.shuffle(index)
        data = X1[index, :, :, :]
        label = Y1[index]
        print("打乱数据维度", data.shape)
        print("打乱标签维度", label.shape)
    return data, label


class SignalDataset2D_fewshot(Dataset):
    """数据加载器"""

    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.X, self.Y = load_h5_2D(data_folder)  # (3392, 8192, 1)
        self.Y1 = self.Y.reshape(-1)
        self.Y1 = [int(y) for y in self.Y1]
        self.df = pd.DataFrame(self.Y1)
        # self.df.columns = ['class_id', 'id']
        self.df = self.df.assign(id=self.df.index.values)
        self.df.columns = ['class_id', 'id']

    def __getitem__(self, item):
        # 返回一个音频数据
        X = self.X[item]
        Y = self.Y[item]
        return X, Y

    def __len__(self):
        return len(self.X)


if __name__ == '__main__':
    pass
