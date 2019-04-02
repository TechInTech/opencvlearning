# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np

# ********************* 边缘保留滤波(EPF) **********************
"""
实现方法：
1、高斯双边
2、均值迁移
"""


# 高斯双边
def bilater(image):
    dst = cv.bilateralFilter(image, 0, 100, 15)
    cv.imshow('bilater', dst)


#均值迁移
def shift(image):
    dst = cv.pyrMeanShiftFiltering(image, 10, 50)
    cv.imshow('shift image', dst)


def main():

    """
    ../ 表示当前文件所在的目录的上一级目录
    ./ 表示当前文件所在的目录(可以省略)
    / 表示当前站点的根目录(域名映射的硬盘目录)
    """
    src = cv.imread('../picsrc/cai.jpg')
    cv.namedWindow('People', cv.WINDOW_AUTOSIZE)
    cv.imshow('cai', src)

    t1 = cv.getTickCount()

    # 高斯双边
    bilater(src)

    # 均值滤波
    shift(src)


    t2 = cv.getTickCount()
    time = (t2 - t1)/cv.getTickFrequency()
    print('Time consume: %f s' % time)


    cv.waitKey(0)  # 程序等待指定的时间后执行以下语句(时间单位为：ms)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
