from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet

def train_val_data_process(): # 处理训练集和验证集 train训练 val验证
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              # 归一化处理数据集
                              download=True)

    train_data,val_data = Data.random_split(train_data,[round(0.8*len(train_data)),round(0.2*len(train_data))]) # 划分80%的训练集和20%验证集
    # 数据打包
    train_dataloader = Data.DataLoader(train_data, batch_size=128, shuffle=True,num_workers=8)
    val_dataloader = Data.DataLoader(val_data, batch_size=128, shuffle=True,num_workers=8)

    return train_dataloader, val_dataloader