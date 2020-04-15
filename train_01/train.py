''
'创建训练器--以训练三个网络'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sampling import FaceDataset
import os
import net


'创建训练器'
class Trainer:
    ''
    '训练器的参数为 网络，训练数据路径，参数的保存路径，'
    def __init__(self, net, dataset_path,save_path , isCuda=True):
        self.net = net
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.isCuda = isCuda

        if self.isCuda:
            self.net.cuda()

        '创建损失函数：置信度损失'
        '''二分类交叉熵损失函数，是多分类交叉熵(CrossEntropyLoss)的一个特例，
        用BCELoss前面必须用sigmoid激活，用CrossEntropyLoss前面必须用softmax函数'''
        self.cls_loss_fn = nn.BCELoss()

        '偏移量损失'
        self.offset_loss_fn = nn.MSELoss()

        '创建优化器'
        self.optimer = optim.Adam(self.net.parameters())

        '恢复网络训练--加载模型参数，继续训练'
        try:
        # if os.path.exists(self.save_path): # 如果文件存在，接着继续训练
            net.load_state_dict(torch.load(save_path))
            print('加载成功')
        except:
            print('不成功')

    '训练方法'

    def train(self):
        ''
        '导入数据集'
        faceDataset = FaceDataset(self.dataset_path)
        '''数据加载器 ， （num_workers=4，有4个线程加载数据(加载数据需要时间，以防空置)
        drop_last:为True时，防止批次不足报错  '''
        dataloader = DataLoader(faceDataset,batch_size=64, shuffle=True, num_workers=1, drop_last=True)

        while True:
            '从数据集中取出图片，置信度，偏移量'
            for i, (img_data_, category_, offset_) in enumerate(dataloader):
                if self.isCuda:
                    img_data_ = img_data_.cuda()
                    category_ = category_.cuda()
                    offset_ = offset_.cuda()

                '数据集中的图片，从网络输出 置信度，偏移量'
                _output_category, _output_offset = self.net(img_data_)

                # '[512, 1, 1, 1]'
                # # print('置信度', _output_category.shape)
                # '[512, 1, 1, 4]'
                # # print('偏移量', _output_offset.shape)

                '原始图片置信度改变后的形状, [512, 1]'
                output_category = _output_category.view(-1, 1)
                # print('置信度改变后的形状',output_category.shape)

                '偏移量改变后的形状，[512, 4]'
                output_offset = _output_offset.view(-1, 14)
                # print('偏移量改变后的形状', output_offset)
                # print('第一轮计算损失完成')
                # print(output_offset.shape)
                '计算分类的损失--置信度'

                '''处理后去掉部分样本的  置信度(标签) '''


                '取置信度小于2的样本，是布尔值'
                category_mask = torch.lt(category_, 2)

                category = torch.masked_select(category_, category_mask)

                output_category = torch.masked_select(output_category, category_mask)



                '原始置信度和去掉部分样本的置信度作比较，得到处理后的置信度'
                ''

                # print('1原始置信度和去掉部分样本的置信度作比较，得到处理后的置信度',category.shape)


                # print('2原始置信度和去掉部分样本的置信度作比较，得到处理后的置信度',output_category)

                '网络输出的置信度和 标签处理后的置信度作损失计算，得出置信度损失'
                cls_loss = self.cls_loss_fn(output_category, category)

                '''计算bound回归的损失--偏移量，
                    去掉负样本'''
                offset_mask = torch.gt(category_, 0)

                '筛先出非负样本的索引'
                offset_index = torch.nonzero(offset_mask)[:, 0]

                '标签里偏移量'
                offset = offset_[offset_index]

                # offset_offset=torch.masked_select(_output_offset,offset_mask)
                # offset=torch.masked_select(offset_,offset_mask)


                '输出的偏移量'
                output_offset = output_offset[offset_index]

                '偏移量损失'
                # print(offset_offset.shape)
                # print(offset.shape)
                offset_loss = self.offset_loss_fn(output_offset, offset)

                '总损失:置信度损失 + 偏移量损失'
                loss = cls_loss + offset_loss

                '反向传播，优化网络'
                '清空之前的梯度'
                self.optimer.zero_grad()
                '计算梯度'
                loss.backward()
                '优化网络'
                self.optimer.step()
                # print(loss)

                print("i=", i, "loss:", loss.cpu().data.numpy(), "cls_loss:",cls_loss.cpu().data.numpy(), " offset_loss",
                      offset_loss.cpu().data.numpy())

# 保存
                if (i+1)%1000==0:
                    'state_dict()是保存参数'
                    torch.save(self.net.state_dict(), self.save_path) # state_dict保存网络参数，save_path参数保存路径
                    print("save success")                            # 每轮次保存一次；最好做一判断：损失下降时保存一次






















