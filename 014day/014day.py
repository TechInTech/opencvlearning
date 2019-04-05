# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ********************* 图像金字塔原理及其应用 **********************
"""
1、PyrDown 降采样
2、PyrUp 还原
3、高斯金字塔与拉普拉斯金字塔
"""

# 高斯金字塔
def pyramid(image):
    level = 4
    temp = image.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow('pyramid_images'+str(i), dst)
        temp = dst.copy()
    return pyramid_images


# 拉普拉斯金字塔
def lapalian(image):
    """
    image的尺度应为：高等于宽
    """
    pyramid_images = pyramid(image)
    level = len(pyramid_images)
    for i in range(level - 1, -1, -1):
        if (i - 1) < 0:
            expand = cv.pyrUp(pyramid_images[i], dstsize = image.shape[:2])
            lpls = cv.subtract(image, expand)
            cv.imshow('lpls'+str(i), lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize = pyramid_images[i-1].shape[:2])
            lpls = cv.subtract(pyramid_images[i-1], expand)
            cv.imshow('lpls'+str(i), lpls)

def main():
    src = cv.imread('../picsrc/pickcai.jpg')
    cv.namedWindow('pic', cv.WINDOW_AUTOSIZE)
    cv.imshow('pic', src)

    t1 = cv.getTickCount()

    # pyramid(src)

    lapalian(src)

    t2 = cv.getTickCount()
    time = (t2 - t1)/cv.getTickFrequency()
    print('Time consume: %f s' % time)

    cv.waitKey(0)  # 程序等待指定的时间后执行以下语句(时间单位为：ms)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
