# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np

# ********************* Canny边缘提取及其应用 **********************
"""
Canny算法步骤：
1、高斯模糊 -- GaussianBlur (Canny 对噪声比较敏感，故需要高斯模糊去噪)
2、灰度转换 -- cvtColor
3、计算梯度 -- Sobel/Scharr
4、非最大信号抑制
5、高低阈值输出二值函数  -- threshold
"""

# Canny边缘提取
def edge(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_RGB2GRAY)

    # xgradient
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)

    #ygradient
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)

    # edge
    edge = cv.Canny(xgrad, ygrad, 50, 150)
    # or
    # edge = cv.Canny(gray, 50, 150)

    dst = cv.bitwise_and(image, image, mask = edge)
    cv.imshow('color Edge', dst)


def main():
    src = cv.imread('../picsrc/cai.jpg')
    cv.namedWindow('pic', cv.WINDOW_AUTOSIZE)
    cv.imshow('template', src)

    t1 = cv.getTickCount()

    edge(src)

    t2 = cv.getTickCount()
    time = (t2 - t1)/cv.getTickFrequency()
    print('Time consume: %f s' % time)

    cv.waitKey(0)  # 程序等待指定的时间后执行以下语句(时间单位为：ms)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
