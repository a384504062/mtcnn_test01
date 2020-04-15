''
print('''造24*24训练样本的思路
目的：
需要储存文件的文件夹的文件，此文件包函.jpg图片文件和.txt的文档文件
需要原celebA的数据原图文件和标签文件

''')

print("导入相关模块")
import os
import traceback
import numpy as np
import utils
import PIL.Image as Image

# print('需要原图的图片和标签的路径')
anno_src = r"E:\MTCNN\MTCNN\cebela\Anno\list_bbox_celeba.txt"
# print('已创建变量，储存celebA的图片 路径')
img_dir = r"E:\MTCNN\MTCNN\cebela\img_celeba"

save_path = r"E:\MTCNN\MTCNN\June\cebela_04"
# print('已创建变量，储存已处理好的文件夹 的路径')

# print('生成尺寸为12的新人脸的样本，12的人脸样本包函正样本，部分样本，非人脸样本')
for face_size in [24]:
    positive_image_dir = os.path.join(save_path, str(face_size), 'positive')
    negative_image_dir = os.path.join(save_path, str(face_size), 'negative')
    part_image_dir = os.path.join(save_path, str(face_size), 'part')

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    # print('已创建储存24*24的图片所需要的文件夹')

    positive_anno_filename = os.path.join(save_path,str(face_size),"positive.txt")
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")
    # print('已创建储存24*24的图片的标签 所需要的.txt的路径')

    positive_count = 0
    negative_count = 0
    part_count = 0
    # print("生成多少个文件夹的计数器")

    try:
        positive_anno_file = open(positive_anno_filename, 'w')
        negative_anno_file = open(negative_anno_filename, 'w')
        part_anno_file = open(part_anno_filename, 'w')
        # print('已try生成并打开标签的.txt文件，以便可以随时写入新的标签数据')
        # print('开始处理celebA的标签')
        for i, line in enumerate(open(anno_src)):
            if i < 2:
                continue
            try:
                # print('把原始标签中的元素全切')
                strs = line.strip().split()
                # print("这是最好的",strs)
                # print('并取标签中第一个元素的切片')
                image_filename = strs[0].strip()
                # print('并做成原始图片的绝对路径,把标签的第1个元素和储存图片的文件夹')
                image_file = os.path.join(img_dir, image_filename)
                print(image_file)

                # print('可以使用原始图片了，打开并处理原始图片 ')
                with Image.open(image_file) as img:
                    img_w, img_h = img.size
                    # print('再把原始标签中的元素全切，各取不同的索引作为x和y')
                    x1 = float(strs[1].strip())
                    y1 = float(strs[2].strip())
                    w = float(strs[3].strip())
                    h = float(strs[4].strip())
                    x2 = float(x1 + w)
                    y2 = float(y1 + h)

                    # print('排除人脸小于40，和坐标有负值的数据')
                    if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue

                    # print('因为原始框的标注偏大，所以应适当偏移')
                    x1 = int(x1 * 0.12)
                    y1 = int(y1 + h * 0.1)
                    x2 = int(x1 + w * 0.9)
                    y2 = int(y1 + h * 0.85)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    boxes = [[x1, y1, x2, y2]]

                    # print('并计算出偏移后的框的   新的中心点的位置')
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    # print('使样本根据中心点随机偏移')
                    for _ in range(5):
                        '让标准标签中心点有少许偏移'
                        w_ = np.random.randint(-w * 0.2, w * 0.2)
                        h_ = np.random.randint(-h * 0.2, h * 0.2)
                        cx_ = cx + w_
                        cy_ = cy + h_

                        '让人脸形成正方形'
                        side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                        x1_ = np.max(cx_ - side_len / 2, 0)
                        y1_ = np.max(cy_ - side_len / 2, 0)
                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len

                        '得出偏移后的新正方形框'
                        crop_box = np.array([x1_, y1_, x2_, y2_])

                        '最终校正计算出相对于真实框的的偏移量，再归一化框'
                        offset_x1 = (x1 - x1_) / side_len
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len

                        '剪下校正后的人脸，并进行大小缩放'
                        face_crop = img.crop(crop_box)
                        face_resize = face_crop.resize((face_size, face_size),Image.ANTIALIAS)

                        '把剪下扣出的图作IOU计算，分别得出不同的值'
                        iou = utils.iou(crop_box, np.array(boxes))[0]
                        '以上是对原始标签、图像的处理'


                        '以下是对扣出图片的保存和标签的保存'
                        '正样本'
                        if iou > 0.6:
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5} \n".format(
                                    positive_count, 1,
                                    offset_x1, offset_y1,
                                    offset_x2, offset_y2
                                )
                            )
                            positive_anno_file.flush()
                            face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                            positive_count += 1
                        elif iou > 0.4:
                            part_anno_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5} \n".format(
                                    part_count, 2,
                                    offset_x1, offset_y1,
                                    offset_x2, offset_y2
                                )
                            )
                            part_anno_file.flush()
                            face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            part_count += 1
                        elif iou < 0.29:
                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0 \n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1

                        '重新生成新的负样本'
                        _boxes = np.array(boxes)

                    for i in range(5):
                        side_len = np.random.randint(face_size, min(img_w, img_h) / 2)
                        x_ = np.random.randint(0, img_w - side_len)
                        y_ = np.random.randint(0, img_h - side_len)
                        crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])

                        if np.max(utils.iou(crop_box, _boxes)) < 0.29:
                            face_crop = img.crop(crop_box)
                            face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                            negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0 \n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1


            except Exception as e:
                traceback.print_exc()
    #print('关闭写入文件'):
    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()

















