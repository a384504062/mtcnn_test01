import torch
import torchvision.datasets
from torch.utils.data import Dataset
from torch.utils import data
import os
import numpy as np
import PIL.Image as Image
from torchvision import transforms


data_TF = transforms.Compose([
    transforms.ToTensor()
])

class FaceDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.dataset = []
        self.dataset.extend(open(os.path.join(path, 'positive.txt')).readlines())
        self.dataset.extend(open(os.path.join(path, 'negative.txt')).readlines())
        self.dataset.extend(open(os.path.join(path, 'part.txt')).readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        ''
        '切割标签的元素'
        strs = self.dataset[item].strip().split()
        # print('元素',strs)
        '取出置信度'
        cond = torch.Tensor([int(strs[1])])
        # print('置信度',cond)
        '取出偏移量'
        offset = torch.Tensor([float(strs[2]),float(strs[3]),float(strs[4]),float(strs[5]),float(strs[6]),float(strs[7]),float(strs[8]),float(strs[9]),float(strs[10]),float(strs[11]),float(strs[12]),float(strs[13]),float(strs[14]),float(strs[15])])
        # print('偏移量',offset)

        "定义图片的绝对路径"
        img_path = os.path.join(self.path, strs[0])
        # print('图片的绝对路径',img_path)
        '把图片去均值和归一化后再转为张量'
        # img_data = torch.Tensor(np.array(Image.open(img_path))/255 - 0.5)
        # print(img_data)
        img_data = data_TF(Image.open(img_path))
        # print('图片',img_data)

        '已交换图片的轴了'
        # img_data = img_data.permute(2, 0, 1)


        return img_data, cond, offset

if __name__ == '__main__':
    path = r"E:\MTCNN\MTCNN\June\cebela_06\48"  # 只以尺寸为48的为例
    DATA=FaceDataset(path)
    dataset = data.DataLoader(DATA,batch_size=10,shuffle=True)
    for i, (img_data_, category_, offset_) in enumerate(dataset):
        print(img_data_.shape)
        print(category_.shape)
        print(offset_.shape)
    # path = r"F:\MTCNN\June\cebela_02\48"
    # FaceDataset(path)
    # print(a)

    dataset = FaceDataset(path)
    dataset[0]
    print(dataset[0])
