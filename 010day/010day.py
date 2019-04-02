# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ********************* 图像直方图反向投影及其应用 **********************
"""
直方图反向投影
1、HSV与RGB色彩空间
2、反向投影
"""

# 反向投影
def back_projection(sample, target):
    """
    sample:局部图
    target:完整图
    """
    roi_hsv = cv.cvtColor(sample, cv.COLOR_RGB2HSV)
    target_hsv = cv.cvtColor(target, cv.COLOR_RGB2HSV)

    cv.imshow('sample', sample)
    cv.imshow('target', target)

    roiHist = cv.calcHist([roi_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256]) # [32, 32]可以调整，参数小时，效果好
    cv.normalize(roiHist, roiHist, 0, 255, cv.NORM_MINMAX)
    dst = cv.calcBackProject([target_hsv], [0, 1], roiHist, [0, 180, 0, 256], 1)
    cv.imshow('backprojecction', dst)

#
def hist2d(image):
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    hist = cv.calcHist([image], [0, 1], None, [180, 256], [0, 180, 0, 256])
    plt.imshow(hist, interpolation='nearest')
    plt.title('2D Histogram')
    plt.show()

def main():
    src = cv.imread('../picsrc/pickcai.jpg')
    print(src.shape)
    cv.namedWindow('pic', cv.WINDOW_AUTOSIZE)
    # cv.imshow('cai', src)

    t1 = cv.getTickCount()

    src1 = cv.imread('../picsrc/cai.jpg')
    print(src1.shape)
    # cv.imshow('tree', src1)

    # hist2d(src)

    back_projection(src, src1)

    t2 = cv.getTickCount()
    time = (t2 - t1)/cv.getTickFrequency()
    print('Time consume: %f s' % time)


    cv.waitKey(0)  # 程序等待指定的时间后执行以下语句(时间单位为：ms)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
