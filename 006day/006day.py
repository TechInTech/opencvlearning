# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


# ********************* 模糊操作 **********************
"""
1、 基于离散卷积
2、 定义好每个卷积核
3、 不同卷积核得到不同的卷积效果
4、 模糊是卷积的一种表象
"""

# 均值模糊
def blur(image):
    dst = cv.blur(image, (5, 5))
    cv.imshow('blur', dst)


# 中值模糊
def median_blur(image):
    dst = cv.medianBlur(image, 5)
    cv.imshow('median_blur', dst)


# 自定义模糊
def custom_blur(image):

    # 自定义模糊
    kernel = np.ones([5, 5], np.float32) / 25
    # kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.float32) / 9   # 轻度模糊

    # 自定义锐化
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) # 锐化算子(提高对比度)

    """
    锐化需满足一定条件
    1、为奇数 ；
    2、总和为0或1:
     0在做边缘、梯度时需要；
     1在做增强时需要；
    """

    dst = cv.filter2D(image, -1, kernel = kernel)
    cv.imshow('custom_blur', dst)

def main():
    src = cv.imread('../picsrc/cai.jpg')
    cv.namedWindow('People', cv.WINDOW_AUTOSIZE)
    cv.imshow('cai', src[])

    ## ******************** 测试模糊算法 ***************

    blur(src)

    median_blur(src)

    custom_blur(src)

    ## ************************************************

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
