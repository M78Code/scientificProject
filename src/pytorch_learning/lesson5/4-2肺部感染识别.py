# 1. 加载库
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np


def image_show(inp, title=None):
    plt.figure(figsize=(14, 3))
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()


def main():
    # 3 定义超参数
    BATCH_SIZE = 8  # 每批处理的数据数量
    DEVICE = torch.device("mps")

    # 4.1 构建pipeline, 对图像做处理
    pipeline = transforms.Compose([
        transforms.Resize(300),  # 图片缩小到300
        transforms.RandomResizedCrop(300),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(256),  # 中心裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),  # 对图片进行正则化
    ])

    # 4.2 图片转换
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize(300),  # 图片缩小到300
            transforms.RandomResizedCrop(300),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(256),  # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),  # 对图片进行正则化
        ]),

        "val": transforms.Compose([
            transforms.Resize(300),  # 图片缩小到300
            transforms.CenterCrop(256),  # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),  # 对图片进行正则化
        ])}
    # 5 操作数据集
    # 5.1 加载数据集
    data_path = "/Users/leon/LeonProject/scientificProject/dataset/chest_xray"
    # 5.2 加载数据集train 和 val
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x]) for x in ['train', 'val']}
    # 5.3 给数据集创建一个迭代器，读取数据
    dataloaders = {x: DataLoader(image_datasets[x], shuffle=True, batch_size=BATCH_SIZE) for x in ['train', 'val']}

    # 5.4 训练和验证集的大小（图片的数量）
    # data_size = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # 5.5 获取标签类别的名称: NORMAL 正常， PNEUMONIA 感染
    target_names = image_datasets['train'].classes

    # 6 显示一个batch_size的图片（8张图片）
    # 6.1 读取8张图片
    datas, targets = next(iter(dataloaders['train']))
    # 6.2 将若干张图片拼成一幅图像
    out = make_grid(datas, nrow=4, padding=10)
    # 6.3 显示图片
    image_show(inp=out, title=[target_names[x] for x in targets])


main()


