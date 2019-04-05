# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np

# ********************* 图像形态学及其应用 **********************
"""
形态学处理的核心就是定义结构元素

1、膨胀
2、腐蚀
3、应用：
   -> 开操作
   -> 闭操作
   作用： 1、去除小的干扰块(开操作)
         2、填充闭合区域(闭操作)
         3、水平或者垂直线提取 *****
4、其它形态学操作
   > 顶帽
   > 黑猫
   > 形态学梯度
"""

# 腐蚀
def erode_img(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow('binary', binary)

    # 形态学处理的核心就是定义结构元素 cv.MORPH_RECT 定义矩形
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.erode(binary, kernel)
    cv.imshow('erode', dst)


# 膨胀
def dilate_img(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow('binary', binary)

    # 形态学处理的核心就是定义结构元素
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.dilate(binary, kernel)
    cv.imshow('dilate', dst)


# 开操作
def open_do(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow('binary', binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    cv.imshow('open image', binary)

# 闭操作
def close_do(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow('binary', binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    cv.imshow('open image', binary)


# 二值帽操作
def hat_binary(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    dst = cv.morphologyEx(binary, cv.MORPH_BLACKHAT, kernel)
    cv.imshow('hat_binary', dst)

# 灰度帽操作
def hat_gray(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15)) # 参数可变
    dst = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)
    cimage = np.array(gray.shape, np.uint8)
    cimage = 100
    dst = cv.add(dst, cimage)
    cv.imshow('hat_gray', dst)

# 基本梯度操作
def basic_gradient(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(binary, cv.MORPH_GRADIENT, kernel)
    cv.imshow('basicGrandient', dst)

# 内外梯度操作
def gradient(image):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dm = cv.dilate(image, kernel)
    em = cv.erode(image, kernel)
    dst1 = cv.subtract(image, em) # internal gradient
    dst2 = cv.subtract(dm, image) # external gradient
    cv.imshow('internal', dst1)
    cv.imshow('external', dst2)


def main():
    src = cv.imread('../picsrc/cai.jpg')
    cv.namedWindow('pic', cv.WINDOW_AUTOSIZE)
    cv.imshow('template', src)

    t1 = cv.getTickCount()

    # 腐蚀方法
    # erode_img(src)

    # 膨胀
    # dilate_img(src)

    # 开、闭操作
    # open_do(src)
    # close_do(src)


    # 顶帽、黑帽操作(二值)
    # hat_binary(src)

    # # 顶帽、黑帽操作(灰度)
    # hat_gray(src)
    #
    # # 基本梯度
    # basic_gradient(src)
    #
    # # 内、外梯度
    gradient(src)

    t2 = cv.getTickCount()
    time = (t2 - t1)/cv.getTickFrequency()
    print('Time consume: %f s' % time)

    cv.waitKey(0)  # 程序等待指定的时间后执行以下语句(时间单位为：ms)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
