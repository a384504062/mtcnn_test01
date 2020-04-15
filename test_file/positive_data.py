'王灵珍给的生成正样本的代码'
import os
from PIL import Image
import numpy as np
# from test_20190628.tool import utils
import utils

anno_src = r"E:\MTCNN\MTCNN\cebela\Anno\list_bbox_celeba.txt"
landmarks_src = r"E:\MTCNN\MTCNN\cebela\Anno\list_landmarks_celeba.txt"
img_dir = r"E:\MTCNN\MTCNN\cebela\img_celeba"

save_path = r"E:\MTCNN\MTCNN\June\cebela_05"

for face_size in [48]:
    print("gen %i image" % face_size)
    # 样本图片存储路径
    positive_image_dir = os.path.join(save_path, str(face_size), "positive_landmark")
    if not os.path.exists(positive_image_dir):
        os.makedirs(positive_image_dir)
    # 样本描述存储路径
    positive_anno_filename = os.path.join(save_path, str(face_size), "positive_landmark.txt")


    # 样本数量、样本数量开关
    num_flag = True
    flag = False
    positive_count = 0

    # 文件操作try一下、防止程序崩溃
    try:
        positive_anno_file = open(positive_anno_filename,"w")     # 以写入的方式打开文件
        while num_flag: # 样本数量开关
            filePath = []
            coordi = []
            landmark = []
            info = []

            print('* GETTING FILES...')
            coordi.extend(open(anno_src).readlines())
            landmark.extend(open(landmarks_src).readlines())
            for idx in range(len(coordi)):
                if flag:
                    break
                if idx < 2:
                    continue

                box = []
                line1 = coordi[idx].split()
                line2 = landmark[idx].split()
                filePath.append(line1[0])  # 图片名称

                box.extend(line1[1:])
                box.extend(line2[1:])
                info.append(box)  # 四个coordi，十个landmark

            print('* DATA GENERATING...')
            # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for 循环当中。
            for no, (name, info) in enumerate(zip(filePath, info)):  # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
                # if no==10:
                #     break
                img = Image.open(os.path.join(img_dir, name))
                w,h=int(info[2]),int(info[3])
                x1, y1, x2, y2 = int(info[0]), int(info[1]), int(info[0]) + int(info[2]), int(info[1]) + int(info[3])  # coordi
                lex, ley, rex, rey, nx, ny, lmx, lmy, rmx, rmy = int(info[4]), int(info[5]), int(info[6]), int(info[7]), int(info[8]), int(info[9]), int(info[10]), int(info[11]), int(info[12]), int(info[13])  # landmark


                # 过滤较小的框，降低误框率
                if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                    continue

                boxes = [[x1, y1, x2, y2]]
                # 计算出人脸中心点位置
                cx = x1 + w / 2
                cy = y1 + h / 2

                #使正样本和部分样本数量翻倍
                for _ in range(5):
                    #让人脸中心点有少许的偏移
                    w_ = np.random.randint(-int(w * 0.2), int(w * 0.2) + 2)  # 中心点移动的距离
                    h_ = np.random.randint(-int(h * 0.2), int(h * 0.2) + 2)
                    cx_ = cx + w_
                    cy_ = cy + h_

                    # 让人脸形成正方形，并且让坐标也有少许的偏离
                    side_len = np.random.randint(int(min(w, h) * 0.6), int(max(w, h) * 1.5) + 2)
                    x1_ = np.max(cx_ - side_len / 2, 0)
                    y1_ = np.max(cy_ - side_len / 2, 0)
                    x2_ = x1_ + side_len
                    y2_ = y1_ + side_len

                    crop_box = np.array([x1_, y1_, x2_, y2_])

                    # 计算坐标的偏移值
                    offset_x1 = (x1 - x1_) / side_len
                    offset_y1 = (y1 - y1_) / side_len
                    offset_x2 = (x2 - x2_) / side_len
                    offset_y2 = (y2 - y2_) / side_len

                    offset_px1 = (lex - x1_) / side_len
                    offset_py1 = (ley - y1_) / side_len
                    offset_px2 = (rex - x1_) / side_len
                    offset_py2 = (rey - y1_) / side_len
                    offset_px3 = (nx - x1_) / side_len
                    offset_py3 = (ny - y1_) / side_len
                    offset_px4 = (lmx - x1_) / side_len
                    offset_py4 = (lmy - y1_) / side_len
                    offset_px5 = (rmx - x1_) / side_len
                    offset_py5 = (rmy - y1_) / side_len

                    # 剪切下图片，并进行大小缩放
                    face_crop = img.crop(crop_box)
                    face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                    iou = utils.iou(crop_box, np.array(boxes))[0]
                    if iou > 0.7:  # 正样本
                        positive_anno_file.write("positive_landmark/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(positive_count, 1, offset_x1, offset_y1,
                             offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2,offset_px3,offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                        positive_anno_file.flush()
                        face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                        positive_count += 1
                        if positive_count == 60000:
                            num_flag = False
                            flag = True
                            break#break语句用在while和for循环中。
                if flag:
                    break#如果您使用嵌套循环，break语句将停止执行最深层的循环，并开始执行下一行代码。

    finally:
        positive_anno_file.close()





