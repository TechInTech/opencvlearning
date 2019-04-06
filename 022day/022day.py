# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np

# ********************* 分水岭算法及其应用 **********************
"""
分水岭算法

1、距离变换
2、分水岭变换介绍

"""


def main():
    src = cv.imread('../picsrc/cai.jpg')
    cv.namedWindow('pic', cv.WINDOW_AUTOSIZE)
    cv.imshow('template', src)

    t1 = cv.getTickCount()

    # 腐蚀方法
    # erode_img(src)

    t2 = cv.getTickCount()
    time = (t2 - t1) / cv.getTickFrequency()
    print('Time consume: %f s' % time)

    cv.waitKey(0)  # 程序等待指定的时间后执行以下语句(时间单位为：ms)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
