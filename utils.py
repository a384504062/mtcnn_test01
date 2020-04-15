import numpy as np

'对原框和偏移框做重叠率的计算'
def iou(box, boxes, isMin=False):
    ''
    '计算原框的面积：[x1,y1,x2,y2]'
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    # print(box_area)

    '偏移的新框的面积'
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # print(area)

    '新的交集框的坐标点，取两个元素中最大的那个元素'
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    '判断两个框是否有交集'
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    '交集框的面积'
    inter = w * h

    '最终判断是否大框套小框，还是两个框为普通的交集框'
    if isMin:
        ovr = np.true_divide(inter, np.minimum(box_area, area))
    else:
        ovr = np.true_divide(inter, (box_area + area - inter))

    return ovr

if __name__ == '__main__':
    a = np.array([1,2,3,4])
    b = np.array([[1,4,2,5],[2,1,3,5]])
    iou = iou(a,b)
    # print(iou)


'非极大值抑制'
'''思路：首先根据对置信度进行排序，找出最大值框与每个框做IOU比较，
再将保留下来的框再进行循环比较，直到符合条件，保留其框'''
def nms(boxes, thresh=0.3, isMin =False):
    ''
    '框的长度为0时(防止程序有缺陷报错)'
    if boxes.shape[0] == 0:
        return np.array([])

    '框的长度不为0时'
    '根据置信度排序：[x1, y1, x2, y2, C]'
    '根据置信度“由大到小”，默认有小到大(加符号可反向排序)'
    _boxes = boxes[(-boxes[:, 4]).argsort()]
    # print(_boxes.shape)

    '创建空列表，存放保留剩余的框'
    r_boxes = []

    '用1st个框，与其余的框进行比较，当长度小于等于1时停止（比len(_boxes)-1次）'
    while _boxes.shape[0] > 1:
        '取出第1个框'
        a_box = _boxes[0]

        '取出剩余的框'
        b_boxes = _boxes[1:]

        '将1st个框加入列表'
        '每循环一次往，添一个框'
        r_boxes.append(a_box)

        '比较IOU, 将符合阈值条件的框保留下来'
        '将阈值小于0.3的建议框保留下来，返回保留框的索引'
        index = np.where(iou(a_box, b_boxes, isMin) < thresh)
        # print(index.shape)
        '循环控制条件，取出阈值小于0.3的建议框'
        _boxes = b_boxes[index]

    '''最后一次，结果只用1st个符合或只有一个符合，
        若框的个数大于1，此处_boxes调用的是whilex 循环里的，
        此判断条件放在循环里和外都可以(只有在函数类外才可产生局部作用于)'''
    if _boxes.shape[0] > 0:
        '将此框添加到列表中'
        r_boxes.append(_boxes[0])

    'stack组装为矩阵：将列表中的数据在0轴上堆叠（行方向）'
    return np.stack(r_boxes)

'扩充：找到中心点，及最大边长，沿着最大边长的两边扩充'
'将长方向框，补齐转成正方形框'
def convert_to_square(bbox):
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])

    '框高'
    h = bbox[:, 3] - bbox[:, 1]

    '框宽'
    w = bbox[:, 2] - bbox[:, 0]

    '返回最大边长'
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + w * 0.5 - max_side * 0.5

    '加最大边长，加最大边长，决定了正方形框'
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side

    return square_bbox

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)

    return y



























