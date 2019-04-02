# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np

# 算术运算
def add_image(m1, m2):
    dst = cv.add(m1, m2)
    cv.imshow('add_image', dst)
    # cv.imwrite('./redwhitetree.png', dst)

def subtract_image(m1, m2):
    dst = cv.subtract(m1, m2)
    cv.imshow('subtract_image', dst)
    # cv.imwrite('./reddarktree.png', dst)

def divide_image(m1, m2):
    dst = cv.divide(m1, m2)
    cv.imshow('divide_image', dst)
    # cv.imwrite('./reddarktree.png', dst)

def multiply_image(m1, m2):
    dst = cv.multiply(m1, m2)
    cv.imshow('multiply_image', dst)
    # cv.imwrite('./reddarktree.png', dst)

# 逻辑运算
def logical_image(m1, m2):
    # dst = cv.bitwise_and(m1, m2)
    # cv.imshow('and_image', dst)
    #
    # dst = cv.bitwise_or(m1, m2)
    # cv.imshow('or_image', dst)

    dst = cv.bitwise_not(m1)
    cv.imshow('not_image', dst)

# 调整对比度、亮度
def contrast_brightness(image, c, b):
    """
    c: 对比度
    b: 亮度
    """
    h, w, ch = image.shape   # 获得图像维度
    blank = np.zeros([h, w, ch], image.dtype)  # 创建空白图像
    dst = cv.addWeighted(image, c, blank, 1 - c, b)
    cv.imshow('con-bri-image', dst)


src1 = cv.imread('hsvimage.png')
src2 = cv.imread('yuvimage.png')

print(src1.shape)
print(src2.shape)

cv.namedWindow('music', cv.WINDOW_AUTOSIZE)
# cv.imshow('hsv', src1)
# cv.imshow('yuv', src2)

t1 = cv.getTickCount()


# 图像的加、减、乘、除算术运算

# add_image(src1, src2)
#
# subtract_image(src1, src2)
#
# divide_image(src1, src2)
#
# multiply_image(src1, src2)

# 逻辑运算
# logical_image(src1, src2)


src3 = cv.imread('cai.jpg')
cv.imshow('cai', src3)

# 增加对比度、亮度
contrast_brightness(src3, 1.2, 1)


t2 = cv.getTickCount()
time = (t2 - t1)/cv.getTickFrequency()
print('Time: %f ms' % time)
cv.waitKey(0)

cv.destroyAllWindows()
