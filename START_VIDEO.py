'openCV'
from Detector import Detector
import cv2
import PIL.Image as pimg
import PIL.ImageDraw as draw
import numpy as np
from tools import *
from torchvision import transforms

video_path = r"E:\My_project\MTCNN\test_vidio\test.mp4"

data_TF = transforms.Compose([
    transforms.ToTensor()
])


def Video(video_path):
    detect=Detector()                              # 实例化detector
    video=cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # 定义编解码器并创建VideoWriter对象，即输入四个字符代码即可得到对应的视频编码器（XVID编码器）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    vout = cv2.VideoWriter()
    vout.open(r'E:\My_project\MTCNN\test_vidio\video1.mp4', fourcc, fps, size, True)
    count=0
    while True:
        ret,frame=video.read()
        if not ret:
            print("--")
            break
        # vout.write(frame)
        img = pimg.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if count%4==0:
            re_boxes=detect.net_detect(img)     # 获得人脸位置和关键点信息
        pen=draw.ImageDraw(img)
        colorSet=0
        for box in re_boxes:
            x1,y1,x2,y2,_=box
            pen.rectangle(xy=(x1,y1,x2,y2),outline=(255,127,colorSet),width=1)
        frame=np.array(img)[:,:,[2,1,0]]
        vout.write(frame)
        cv2.imshow('FACE CATCHER',frame)
        cv2.waitKey(1)
        count+=1
    vout.release()
    cv2.destroyAllWindows()

# fourcc = cv2.VideoWriter_fourcc(*'XVID')  #定义编解码器并创建VideoWriter对象，即输入四个字符代码即可得到对应的视频编码器（XVID编码器）
# out = cv2.VideoWriter('video1.avi', fourcc, 20.0, (640,480))  #参数：保存文件名，编码器，帧率，视频宽高
# #判断视频流读取是否正确
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         frame = cv2.flip(frame, 0)  #flip（）函数是进行图像翻转：1：水平翻转，0：垂直翻转，-1：水平垂直翻转
#         out.write(frame)  #保存录像结果
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
# cap.release()
# out.release()
# cv2.destroyAllWindows()


if __name__ == '__main__':
    Video(video_path)

