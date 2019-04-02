# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ********************* 图像二值化及其应用 **********************
"""
1、二值图像

2、图像二值化方法

   -->全局阈值：
      ---->OTSU
      ---->Triangle
      ---->自动与手动

   -->局部阈值：
      ----> cv.ADAPTIVE_THRESH_MEAN_C
      ----> cv.ADAPTIVE_THRESH_GAUSSIAN_C
"""

#全局阈值
def threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    # ret, binary = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
    print('threshold value: %s' % ret)
    cv.imshow('binary', binary)

# 局部阈值(自适应阈值)
def local_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)
    # binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
    cv.imshow('local binary', binary)


# 自定义阈值方法
def custom_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    h, w = image.shape[:2]
    m = np.reshape(gray, [1, h * w])
    mean = m.mean()
    print('mean:', mean)
    ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    cv.imshow('custom threshold', binary)


def main():
    src = cv.imread('../picsrc/cai.jpg')
    print(src.shape)
    cv.namedWindow('pic', cv.WINDOW_AUTOSIZE)
    cv.imshow('cai', src)

    t1 = cv.getTickCount()

    # threshold(src)

    # local_threshold(src)

    custom_threshold(src)

    t2 = cv.getTickCount()
    time = (t2 - t1)/cv.getTickFrequency()
    print('Time consume: %f s' % time)


    cv.waitKey(0)  # 程序等待指定的时间后执行以下语句(时间单位为：ms)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
