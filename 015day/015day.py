# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np

# ********************* 图像梯度及其应用 **********************
"""
1、一阶导数与Sobel算子
2、Sobel算子
3、二阶导数
4、拉普拉斯算子
"""

def lapalian(image):

    # 内置api
    dst = cv.Laplacian(image, cv.CV_32F)
    lpls = cv.convertScaleAbs(dst)

    # 自定义API
    # kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # # kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    # dst = cv.filter2D(image, cv.CV_32F, kernel = kernel)
    # lpls = cv.convertScaleAbs(dst)

    cv.imshow('lapalian', lpls)

def sobel(image):
    # ## Soble算子
    # grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)
    # grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)

    ## Scharr算子(Soble增强版)(当图像边缘很弱，Soble得不到强烈的边缘时，Scharr可以获得增强边缘)
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)

    gradx = cv.convertScaleAbs(grad_x)
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow('gradient-x', gradx)
    cv.imshow('gradient-y', grady)

    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow('gradient', gradxy)


def main():
    src = cv.imread('../picsrc/cai.jpg')
    cv.namedWindow('pic', cv.WINDOW_AUTOSIZE)
    cv.imshow('template', src)

    t1 = cv.getTickCount()

    sobel(src)

    # lapalian(src)

    t2 = cv.getTickCount()
    time = (t2 - t1)/cv.getTickFrequency()
    print('Time consume: %f s' % time)

    cv.waitKey(0)  # 程序等待指定的时间后执行以下语句(时间单位为：ms)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
