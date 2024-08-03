import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import numpy as np
import pandas as pd

# 加载数据集(https://blog.csdn.net/qq_52643100/article/details/140360548)
# train_dataset = datasets.FashionMNIST(root="./data/", train=True, transform=transforms.ToTensor(), download=False)
# test_dataset = datasets.FashionMNIST(root="./data/", train=False, transform=transforms.ToTensor())

import platform

print(platform.platform())
print(platform.architecture())
print(platform.machine())
print(platform.system())


def isMacChip():
    return platform.machine() == "arm64"


print(isMacChip())


class RNN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # 分离隐藏状态，避免梯度爆炸
        out, hn = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out


import os

os.makedirs(os.path.join("..", "data"), exist_ok=True)
data_file = os.path.join("..", "data", "house_label.csv")
with open(data_file, "w") as f:
    f.write("label,house_num\n")
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
inputs, outputs = data.iloc[:, :-1], data.iloc[:, -1].values
inputs = inputs.fillna(inputs.mean())
