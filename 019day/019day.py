# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np

# ********************* 轮廓发现及其应用 **********************
"""
操作步骤：
1.转换图像为二值化图像：threshold方法或者canny边缘提取获取的都是二值化图像
2.通过二值化图像寻找轮廓：findContours
3.描绘轮廓：drawContours
"""

# 获得二值图像


def edge(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    edge = cv.Canny(gray, 30, 50)
    cv.imshow('Canny Edge', edge)
    return edge


def contours(image):
    """
    # 获得二值图像法1
    dst = cv.GaussianBlur(image, (3, 3), 0) #高斯模糊，消除噪声
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY) #先变灰度图像
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)   #获取二值图像
    cv.imshow('binary image', binary)
    """
    # 获得二值图像法2
    binary = edge(image)

    cloneImage, contours, heriachy = cv.findContours(
        binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # RETR_TREE包含检测内部
    # cloneImage, contours, heriachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #RETR_EXTERNAL检测外部轮
    for i, contour in enumerate(contours):
        # cv.drawContours(image, contours, i, (0, 0, 255), -1)  # -1填充轮廓， 2显示轮廓
        cv.drawContours(image, contours, i, (0, 0, 255), 2)  # 绘制轮廓
        print(i)
    cv.imshow('detect contours', image)


def main():
    src = cv.imread('../picsrc/circle.jpg')
    cv.namedWindow('pic', cv.WINDOW_AUTOSIZE)  # 创建GUI窗口,形式为自适应
    cv.imshow('template', src)  # 通过名字将图像和窗口联系

    t1 = cv.getTickCount()

    # 1
    contours(src)

    t2 = cv.getTickCount()
    time = (t2 - t1) / cv.getTickFrequency()
    print('Time consume: %f s' % time)

    cv.waitKey(0)  # 等待用户操作，里面等待参数是毫秒，我们填写0，代表是永远，等待用户操作
    cv.destroyAllWindows()  # 销毁所有窗口


if __name__ == '__main__':
    main()
