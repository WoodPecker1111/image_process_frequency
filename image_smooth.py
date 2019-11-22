# coding=utf-8
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
def readpicture(path):
    im = Image.open(path).convert('L')
    if im is None:
        print("图像打开失败")
        exit()
    return im

# 把图像转换为频域
def transFFT(image):
    # 二维快速傅里叶变换函数
    im = np.fft.fft2(image)
    # 将零频点移到频谱的中间
    fshift = np.fft.fftshift(im)
    # 将复数变换为实数
    # 取log将数据变化到较小的范围
    fshift1 = np.log(abs(fshift))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.subplot(122)
    plt.imshow(fshift1, cmap='gray')
    plt.show()
    return fshift

# 低通滤波器
def lowPassFilter(r, fshift):
    transfor_matrix = np.zeros(fshift.shape)
    center_point = tuple(map(lambda x: (x - 1) / 2, fshift.shape))
    for i in range(transfor_matrix.shape[0]):
        for j in range(transfor_matrix.shape[1]):
            def cal_distance(pa, pb):
                from math import sqrt
                dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                return dis

            dis = cal_distance(center_point, (i, j))
            if dis <= r:
                transfor_matrix[i, j] = 1
            else:
                transfor_matrix[i, j] = 0
    return transfor_matrix


# 将频域转换为图像
def tranPic(image1, image, im):
    '''
    :param image1: 原图像
    :param image: 高斯滤波后的频域图
    :param im: 原图像的频域图
    :return:
    '''
    im = np.abs(np.fft.ifft2(np.fft.ifftshift(image*im)))
    plt.subplot(121)
    # 原图像
    plt.imshow(image1, cmap='gray')
    plt.subplot(122)
    # 变换后图像
    plt.imshow(im, cmap='gray')
    plt.show()


image = readpicture("apple.jpg")
# 返回频域图
im = transFFT(image)
# 对频域平滑
tranImage = lowPassFilter(45, im)
# 返回空域图
tranPic(image, tranImage, im)