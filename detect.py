''
'MTCNN的 测试和使用'
'流程：图像-》缩放-》P网(NMS和边界框回归)-》R网络(NMS和边界框回归)-》O网络(NMS和边界框回归)'

import torch
import net
from torchvision import transforms
import time
import numpy as np
import utils
import os
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw

'网络调参'
'P网络：'
p_cls = 0.9
p_nms = 0.3
'R网络：'
r_cls = 0.8
r_nms = 0.2
'O网络：'
o_cls = 0.97
o_nms = 0.1


'侦测器'
class Detector:
    ''
    '-初始化时加载三个网络的权重(训练好的)，cuda默认设为True-------------------------------------------'
    def __init__(self, pnet_param="./train_01/pnet.pt", rnet_param="./train_01/rnet.pt",
                       onet_param="./train_01/onet.pt",      isCuda=True):
        self.isCuda = isCuda
        '创建实例变量，实例化网络'
        self.pnet = net.PNet()
        self.rnet = net.RNet()
        self.onet = net.ONet()

        if self.isCuda:
            '给网络放到CUDA上运行'
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        '把训练好的权重加载到网络中'
        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))

        '把训练网络里有BN(批归一化时)，要调用eval方法，使用是不用BN, dropout方法'
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        '把图片数据类型转换,归一化及换轴操作'
        self.__image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    '-检测图片----------------------------------------------------------'
    def detect(self, image):
        ''
        'P网络检测-----1st'
        '开始计时'
        start_time = time.time()

        '调用__pnet_detect函数'
        pnet_boxes = self.__pnet_detect(image)
        print('0000000000000',pnet_boxes.shape)
        '若P网络没有人脸时，避免数据出错，返回一个新数组'
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        '计时结束'
        end_time = time.time()
        'P网络所占用的时间差'
        t_pnet = end_time - start_time

        'R网络检测-----2nd'
        start_time = time.time()
        '传入原图，P网络的一些框，根据这些框在原图上抠图'
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        # print("11111111", rnet_boxes.shape)
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time


        'O网络检测-------3rd'
        start_time = time.time()

        '把原图和R网络里的框传到网络中去'
        onet_boxes = self.__onet_detect(image, rnet_boxes)

        '若R网络没有人脸时，避免数据出错，返回一个新数组'
        if onet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_onet = end_time - start_time

        '三个网络检测的总用时间'
        t_sum = t_pnet + t_rnet + t_onet
        print("total:{0} pnet:{1} rnet{2} onet{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return onet_boxes


    '-创建P网络检测函数-------------------------------------------------'
    '-P网络全部都是卷积， 与输入图像的大小无关，可输出任意形状图片'
    def __pnet_detect(self, image):
        ''
        '创建空列表，接收符合条件的建议框'
        boxes = []

        img = image
        w, h = img.size

        '获取图片最小边要去做图像金字塔去缩放，图像金字塔一直缩放到最小边小于或等于12就可以了'
        min_side_len = min(w, h)

        '初始化缩放比例(为1时不缩放)：得到不同分辨率的图片'
        scale = 1

        '最小边大于12时都要缩放'
        while min_side_len > 12:
            '将测试图片数组换轴并转成张量'
            img_data = self.__image_transform(img)
            if self.isCuda:
                '将图片tensor传到cuda中运算'
                img_data = img_data.cuda()

            '在“批次”上升维(测试时传的不止一张图片)'
            'unsqueeze_往第0个位置升维度，把测试图片的维度升为 N C H W 结构，加了个批次 N'
            img_data.unsqueeze_(0)

            '返回多个置信度和偏移量'
            '把图片传到P网络中，输出置信度和偏移量'
            _cls, _offest = self.pnet(img_data)

            '[203, 245]分组卷积特征图的尺寸: W , H '
            cls = _cls[0][0].cpu().data

            '[4, 203, 245] 分组卷积特征图的通道， 尺寸：C, W, H'
            offest = _offest[0].cpu().data


            '''置信度大于0.6的框索引：把P网络输出， 看有没有框到人脸,
            若没有框到人脸，说明网络没有训练好，或者置信度给高了，调低'''
            a = torch.gt(cls, p_cls)
            idxs = torch.nonzero(a)

            '''根据索引，依次添加符合条件的框：cls[idx[0], idx[1]]
             在置信度中取值：idx[0]行索引， idx[1]列索引'''
            for idx in idxs:
                '调用框反算函数_box(把特征图上的框，反算到原图上去)，把大于0.6的框留下来：'
                # print(idx.shape, offest.shape, cls.shape)
                boxes.append(self.__box(idx, offest, cls[idx[0], idx[1]], scale))

            '缩放图片：循环控制条件'
            scale *= 0.7

            '新的宽度'
            _w = int(w * scale)
            _h = int(h * scale)

            '根据缩放后的宽和高，对图片进行缩放'
            img = img.resize((_w, _h))

            '重新获取最小宽高'
            min_side_len = min(_w, _h)

        '尽可能保留IOU小于0.5和一些框下来，若网络训练的好，值可以给低一些'
        print(boxes)
        print(p_cls)
        return utils.nms(np.array(boxes), p_cls)


    '-特征反算：将回归还原到原图上去，根据特征图反算的到原图建议框'
    '-p网络池化步长为2------------------------------------------------------'
    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):
        ''
        '索引乘以步长，除以缩放比例： 特征反算时“行索引，列索引互换”，原为[0]'
        _x1 = (start_index[1].float() * stride) / scale
        _y1 = (start_index[0].float() * stride) / scale
        _x2 = (start_index[1].float() * stride + side_len) / scale
        _y2 = (start_index[0].float() * stride + side_len) / scale

        '人脸所在区域建议框的宽高'
        ow = _x2 - _x1
        oh = _y2 - _y1

        '根据idxs行索引与列索引，找到对应偏移量△δ:[x1,y1,x2,y2]'
        _offset = offset[:, start_index[0], start_index[1]]

        '根据偏移量算实际框的位置，x1=x1_+w*△δ,生样时为：△δ=x1-x1_/w'
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        '正式框：返回4个坐标点和1个偏移量'
        return [x1, y1, x2, y2, cls]


    '-创建R网络检测函数------------------------------------------------------'
    def __rnet_detect(self, image, pnet_boxes):
        ''
        '创建空列表，存放抠图'
        _img_dataset = []

        '给P网络输出的框，找出中心点，沿着最大边长的两边扩充成“正方形”，再抠图'
        _pnet_boxes = utils.convert_to_square(pnet_boxes)

        '遍历每个框，每个框返回4个坐标点，抠图，放缩，数据类型转换，添加列表'
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            '根据4个坐标点枢图'
            img = image.crop((_x1, _y1, _x2, _y2))

            '放缩在固定尺寸'
            img = img.resize((24, 24))

            '将图片数组转成张量'
            img_data = self.__image_transform(img)

            _img_dataset.append(img_data)

        'stack堆叠（默认在0轴），此处相当数据类型转换，'
        img_dataset = torch.stack(_img_dataset)

        if self.isCuda:
            img_dataset = img_dataset.cuda()

        '将27*24的图片传入到网络再进行一次筛选'
        _cls, _offset = self.rnet(img_dataset)

        '将gpu上的数据放到cpu上去，在转成numpy数组'
        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        'R网络要留下来的框，存到boxes中'
        boxes = []

        '''原置信度0.6是偏低的时候很多框并没有用(可打印出来观察)，可以适当调高些, 
        idxs置信度框大于0.6的索引；返回idx:0轴上索引[0, 1], _:1轴上索引[0,0],共同决定元素位置'''
        idxs, _ = np.where(cls > r_cls)

        '根据索引，遍历符合条件的框，1轴上的索引，恰为符合条件的置信度索引(0轴上索引此处用不到)'
        for idx in idxs:
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            '基准框 的高'
            ow = _x2 - _x1
            oh = _y2 - _y1

            '实际框的坐标点'
            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            '返回4个点坐标点和置信度'
            boxes.append([x1, y1, x2, y2, cls[idx][0]])

        '原r_nms为0.5（0.5要往小调），上面的0.6要往大调，小于0.5的框被保留下来'
        return utils.nms(np.array(boxes), r_nms)


    '-创建O网络检测函数-------------------------------------------------------'
    def __onet_detect(self, image, rnet_boxes):
        ''
        '创建列表，存放抠图'
        _img_dataset = []

        '给r网络输出的框，找出中心点，沿着最大边长的两加边扩充成“正方形”'
        _rnet_boxes = utils.convert_to_square(rnet_boxes)

        '''遍历R网络筛选出来的框，计算坐标，抠图，缩放，数据类型转换，添加列表，堆叠'''
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            '根据坐标点“抠图”'
            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48 , 48))

            '将抠出的图转成张量'
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        '堆叠，此处相当于数据格式 转换'
        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.onet(img_dataset)
        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        '存放O网络的计算结果'
        boxes = []

        '''原o_cls为0.97是偏低的，最后要达到标准置信度要达到0.99999，这里可以写成0.99998，这样的话出来就全是人脸，
        留下置 信度大于0.97的框。 返回idx:0轴上索引[0], _:1轴上索引[0],共同决定元素位置。'''
        idxs, _ = np.where(cls > o_cls)

        '根据索引，遍历符合条件的框，1轴上的索引，恰为符合条件的置信度索引（0轴上索引此处用不到）'
        for idx in idxs:
            '以R网络做为基准框'
            _box = _rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            '框的基准宽，框是“方”的，ow = oh'
            ow = _x2 - _x1
            oh = _y2 - _y1

            'O网络最终生成的框的坐标，生样，偏移量△δ=x1-_x1/w*side_len'
            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            '返回4个坐标点和1个置信度'
            boxes.append([x1, y1, x2, y2, cls[idx][0]])

        '用最小面积的IOU，原o_nms(IOU)为小于0.7的框被保留下来'
        return utils.nms(np.array(boxes), o_nms, isMin=True)

if __name__ == '__main__':
    # 多张图片检测
    image_path = r"D:\tools\pycharm\My_PyCharmProjects\MyPyCharm\DeMo_MTCNN_05\mtcnn_test01\test_file\tx.jpg"
    # for i in os.listdir(image_path):
    with torch.no_grad():
        detector = Detector()
        with Image.open(os.path.join(image_path)) as im:  # 打开图片
                # boxes = detector.detect(im)
                # print("----------------------------")
                boxes = detector.detect(im)
                # print(boxes.shape,"ppp")
                # print("size:", im.size)
                imDraw = ImageDraw.Draw(im)
                # print(boxes.shape)
                for box in boxes:  # 多个框，没循环一次框一个人脸
                    # print(boxes)
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])

                    # print((x1, y1, x2, y2))

                    # print("conf:", box[4])  # 置信度
                    imDraw.rectangle((x1, y1, x2, y2), outline='red',width= 6)
        #             # im.show() # 每循环一次框一个人脸
                im.show()
            # exit()

# 备注：以上提到的例子1、2、3见“notes/13-detect”






