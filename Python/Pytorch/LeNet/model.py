import  torch
from torch import nn # 卷积层池化层等函数
from torchsummary import summary # 展示模型参数

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
# 约定俗成三件套
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2) #卷积
        self.sig = nn.Sigmoid() #激活
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2) #池化
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 全连接
        self.flatten = nn.Flatten()  #平展层
        self.f5 = nn.Linear(in_features=16*5*5, out_features=120)
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.f7 = nn.Linear(in_features=84, out_features=10)

    # 前向传播
    def forward(self, x):
        x = self.c1(x)
        x = self.sig(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.sig(x)
        x = self.s4(x)
        x = self.flatten(x)

        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        x = self.sig(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 自动判断cuda是否激活gpu

    # 模型实例化
    model = LeNet().to(device) # 模型放入设备实例化为model
    print(summary(model, (1, 28, 28)))
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 6, 28, 28]             156
           Sigmoid-2            [-1, 6, 28, 28]               0
         AvgPool2d-3            [-1, 6, 14, 14]               0
            Conv2d-4           [-1, 16, 10, 10]           2,416
           Sigmoid-5           [-1, 16, 10, 10]               0
         AvgPool2d-6             [-1, 16, 5, 5]               0
           Flatten-7                  [-1, 400]               0
            Linear-8                  [-1, 120]          48,120
            Linear-9                   [-1, 84]          10,164
           Linear-10                   [-1, 10]             850
          Sigmoid-11                   [-1, 10]               0
================================================================
"""

