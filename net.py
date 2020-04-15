import torch.nn.functional as F
import torch.nn as nn
import torch

'P网络'
class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(10),
            nn.PReLU(),
            # '论文中池化核为2',
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(10,16,kernel_size=3,stride=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),

            nn.Conv2d(16,32,kernel_size=3,stride=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        self.conv4_1 = nn.Conv2d(32,1,kernel_size=1,stride=1)
        self.conv4_2 = nn.Conv2d(32,14,kernel_size=1,stride=1)

    def forward(self, x):
        # print(x.shape)
        x = self.pre_layer(x)

        'P网络 置信度用sigmoid激活(用BCEloss时要先用sigmoid激活)'
        cond = F.sigmoid(self.conv4_1(x))

        'P网络 偏移量不需要激活,原样输出'
        offset = self.conv4_2(x)

        return cond,offset
                                                                # if __name__ == '__main__':
                                                                #     net = PNet()
                                                                #     import torch
                                                                #     xs = torch.randn(size=(3,12,12))
                                                                #     print(net(xs).shape)

'R网络'
class RNet(nn.Module):
    def __init__(self):
        super(RNet,self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=28,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(28),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(28,48,kernel_size=3,stride=1),
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(48,64,kernel_size=2,stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.conv4 = nn.Linear(64*3*3,128)
        self.prelu4 = nn.PReLU()

        'R网络 从池化到全连接'
        self.conv5_1 = nn.Linear(128,1)
        self.conv5_2 = nn.Linear(128,14)


    def forward(self, x):
        print(x.shape)
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)

        'R网络 置信度'
        label = F.sigmoid(self.conv5_1(x))

        'R网络 偏移量'
        offset = self.conv5_2(x)

        return label, offset


'O网络'
class ONet(nn.Module):
    def __init__(self):
        super(ONet,self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(32,64,kernel_size=3,stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),

            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(64,128,kernel_size=2,stride=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.conv5 = nn.Linear(128*3*3, 256)
        self.prelu5 = nn.PReLU()
        self.conv6_1 = nn.Linear(256, 1)
        self.conv6_2 = nn.Linear(256,14)
        # self.conv6_3 = nn.Linear(256, 10)


    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)

        'O网络置信度'
        label = F.sigmoid(self.conv6_1(x))

        'O网络偏移量'
        offset = self.conv6_2(x)

        return label,offset
# if __name__ == '__main__':
#     net=ONet()
#     x=torch.Tensor(10,3,48,48)
#     _,out=net(x)
#     print(out.shape)

