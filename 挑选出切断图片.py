import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 水平方向投影


def hProject(binary):
    h, w = binary.shape

    # 创建h长度都为0的数组
    h_h = [0]*h
    for j in range(h):
        for i in range(w):
            h_h[j] += binary[j, i]
    # h_h 平滑滤波
    h_h = np.array(h_h)
    h_h = np.convolve(h_h, np.ones(10), 'same')
    h_h = h_h.tolist()

    # # 绘制 h_h
    # plt.plot(np.array(h_h))
    # plt.show()

    return h_h

# 垂直反向投影


def vProject(binary):
    h, w = binary.shape
    # 创建 w 长度都为0的数组
    w_w = [0]*w
    for i in range(w):
        for j in range(h):
            w_w[i] += binary[j, i]

    # w_w 平滑滤波
    w_w = np.array(w_w)
    w_w = np.convolve(w_w, np.ones(10), 'same')
    w_w = w_w.tolist()

    # # 绘制 h_h
    # plt.plot(np.array(w_w))
    # plt.show()

    return w_w


def get_image_paths(dir_path: str) -> list[str]:
    '''
    # 获取文件夹和子文件夹内所有图片
    '''
    image_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.bmp') or file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths

# 计算数组头部和尾部的差值


def array_diff(arr: list[int]):
    avg1 = sum(arr[:10])/10
    avg2 = sum(arr[-10:])/10
    return abs(avg1 - avg2) / avg1

# # 用 opencv 计算图片的垂直投影直方图
# if __name__ == '__main__':
#     img = cv2.imread('test.jpg')
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     h_h = hProject(gray)
#     w_w = vProject(gray)
#     print(array_diff(h_h), array_diff(w_w))


imgPaths = get_image_paths("C:\\Unpack\\1300QC_2")
# imgPaths = get_image_paths("C:\\Unpack\\Unpack")
print("{} images found".format(len(imgPaths)))
dst = "C:\\Unpack\\test"
# 创建文件夹
if not os.path.exists(dst):
    os.mkdir(dst)
const_threshold = 0.2
for path in imgPaths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_h = hProject(gray)
    w_w = vProject(gray)
    if array_diff(h_h) > const_threshold or array_diff(w_w) > const_threshold:
        dst_path = os.path.join(dst, os.path.basename(path))
        # # 复制图片到 dst
        # cv2.imwrite(dst_path, img)
        # 移动图片到 dst
        os.rename(path, dst_path)
