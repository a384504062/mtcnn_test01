''
'造12*12训练样本的思路'
import os
import utils
import PIL.Image as Image
import traceback
import numpy as np

'''创建变量，原图的图片和标签的路径'''
img_dir = r"E:\MTCNN\MTCNN\cebela\img_celeba"
anno_src = r"E:\MTCNN\MTCNN\cebela\Anno\anno_landmark.txt"

'''创建变量，储存已处理好的文件夹   的路径'''
save_path = r"E:\MTCNN\MTCNN\June\cebela_06"

# '''创建变量，原图的图片和标签的路径'''
# img_dir = r"F:\MTCNN\CelebA\Img\img_celeba.7z\img_celeba\img_celeba"
# anno_src = r"F:\MTCNN\CelebA\anno_landmark.txt"
#
# '''创建变量，储存已处理好的文件夹   的路径'''
# save_path = r"E:\MTCNN\June\cebela_01"


for face_size in [12]:
    positive_image_dir = os.path.join(save_path, str(face_size), 'positive')
    negative_image_dir = os.path.join(save_path, str(face_size), 'negative')
    part_image_dir = os.path.join(save_path, str(face_size), 'part')

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    print('定义生成多少个文件的计数器')
    positive_count = 0
    negative_count = 0
    part_count = 0


    '定义需要储存48*48图片标签的.txt文件的路径'
    positive_anno_filename = os.path.join(save_path, str(face_size), 'positive.txt')
    negative_anno_filename = os.path.join(save_path, str(face_size), 'negative.txt')
    part_anno_filename = os.path.join(save_path, str(face_size), 'part.txt')

    try:
        '定义并打开.txt文件'
        positive_anno_file = open(positive_anno_filename, 'w')
        negative_anno_file = open(negative_anno_filename, 'w')
        part_anno_file = open(part_anno_filename, 'w')

        '定义48*48具体的图片对象'
        for i, line in enumerate(open(anno_src)):
            if i < 2:
                continue
            try:
                strs = line.strip().split()
                image_filename = strs[0].strip()
                image_file = os.path.join(img_dir, image_filename)

                '打开并处理原始图片'
                with Image.open(image_file) as img:
                    img_w, img_h = img.size
                    x1 = float(strs[1].strip())
                    y1 = float(strs[2].strip())
                    w = float(strs[3].strip())
                    h = float(strs[4].strip())
                    x2 = float(x1 + w)
                    y2 = float(y1 + h)

                    '人的五官'
                    px1 = float(strs[5].strip())
                    py1 = float(strs[6].strip())
                    px2 = float(strs[7].strip())
                    py2 = float(strs[8].strip())
                    px3 = float(strs[9].strip())
                    py3 = float(strs[10].strip())
                    px4 = float(strs[11].strip())
                    py4 = float(strs[12].strip())
                    px5 = float(strs[13].strip())
                    py5 = float(strs[14].strip())



                    '排除人脸小于40，和坐标有负值的数据'
                    if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue

                    '对坐标画出的人脸框进行校正，原框较大'
                    x1 = int(x1 + w * 0.12)
                    y1 = int(y1 + h * 0.1)
                    x2 = int(x1 + w * 0.9)
                    y2 = int(y1 + h * 0.85)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    boxes = [[x1 , y1 , x2 , y2]]

                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    for _ in range(5):
                        # '1.中心点改大一点，就有部分样本'
                        w_ = np.random.randint(-w * 0.05, w * 0.05)
                        h_ = np.random.randint(-h * 0.2, h * 0.2)
                        cx_ = cx + w_
                        cy_ = cy + h_

                        side_len = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
                        x1_ = np.max(cx_ - side_len / 2, 0)
                        y1_ = np.max(cy_ - side_len / 2, 0)
                        x2_ = x1_ + side_len
                        y2_ = y1_ + side_len

                        crop_box = np.array([x1_, y1_, x2_, y2_])

                        '用建议框反算实际框，是衡量建议框和实际框的相对位置，对偏移值进行归一化'
                        offset_x1 = (x1 -x1_) / side_len
                        offset_y1 = (y1 - y1_) / side_len
                        offset_x2 = (x2 - x2_) / side_len
                        offset_y2 = (y2 - y2_) / side_len

                        '五官的偏移值'
                        offset_px1 = (px1 - x1_) / side_len
                        offset_py1 = (py1 - y1_) / side_len
                        offset_px2 = (px2 - x1_) / side_len
                        offset_py2 = (py2 - y1_) / side_len
                        offset_px3 = (px3 - x1_) / side_len
                        offset_py3 = (py3 - y1_) / side_len
                        offset_px4 = (px4 - x1_) / side_len
                        offset_py4 = (py4 - y1_) / side_len
                        offset_px5 = (px5 - x1_) / side_len
                        offset_py5 = (px5 - y1_) / side_len


                        '剪下图片，并进行大小缩放'
                        face_crop = img.crop(crop_box)
                        # print('唯二',face_crop)
                        face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)

                        '作IOU计算'
                        iou = utils.iou(crop_box, np.array(boxes))[0]
                        print(iou)
                        '以上全是对原始标签、图像的处理'

                        '以下是对扣出图片的保存和标签的保存'
                        if iou > 0.6:
                            positive_anno_file.write(
                                'positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n'.format(
                                    positive_count, 1, offset_x1, offset_y1, offset_x2, offset_y2, offset_px1,offset_py1,
                                    offset_px2, offset_py2, offset_px3, offset_py3, offset_px4, offset_py4, offset_px5, offset_py5
                                )
                            )
                            positive_anno_file.flush()
                            face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                            positive_count += 1
                            print(positive_count)
                            # '2.iou 调大一点，就有部分样本'
                    #     elif iou < 0.6:
                    #         part_anno_file.write(
                    #             "part/{0}.jpg {1} {2} {3} {4} {5} \n".format(
                    #                 part_count, 2, offset_x1, offset_y1, offset_x2, offset_y2
                    #             )
                    #         )
                    #         part_anno_file.flush()
                    #         face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                    #         part_count += 0
                    #
                    #     elif iou < 0.29:
                    #         negative_anno_file.write(
                    #             "negative/{0}.jpg {1} 0 0 0 0 \n".format(negative_count, 0))
                    #         negative_anno_file.flush()
                    #         face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                    #         negative_count += 1
                    #
                    #     _boxes = np.array(boxes)
                    #
                    # for i in range(5):
                    #     side_len = np.random.randint(face_size, min(img_w, img_h) / 2)
                    #     x_ = np.random.randint(0, img_w - side_len)
                    #     y_ = np.random.randint(0, img_h - side_len)
                    #     crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])
                    #     # print('唯一',crop_box)
                    #
                    #     if np.max(utils.iou(crop_box, _boxes)) < 0.29:
                    #         face_crop = img.crop(crop_box)
                    #         # print(face_crop)
                    #         face_resize = face_crop.resize((face_size, face_size), Image.ANTIALIAS)
                    #
                    #         negative_anno_file.write("negative/{0}.jpg {1} 0 0 0 0 \n".format(negative_count, 0))
                    #         negative_anno_file.flush()
                    #         face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                    #         negative_count += 1

            except Exception as e:
                traceback.print_exc()

    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()








