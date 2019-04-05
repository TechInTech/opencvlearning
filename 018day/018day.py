# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np

# ********************* 霍夫圆检测及其应用 **********************
"""
"""


# 霍夫圆检测
def detect_circles(image):
    # dst = cv.pyrMeanShiftFiltering(image, 10, 100)
    dst = cv.GaussianBlur(image, (3, 3), 0)
    cimage = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(cimage, cv.HOUGH_GRADIENT, 1, 20, param1 = 90, param2 = 60, minRadius = 1, maxRadius = 0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
    cv.imshow('circle', image)

def main():
    src = cv.imread('../picsrc/circle.jpg')
    cv.namedWindow('pic', cv.WINDOW_AUTOSIZE)
    cv.imshow('template', src)

    t1 = cv.getTickCount()

    # 1
    detect_circles(src)

    t2 = cv.getTickCount()
    time = (t2 - t1)/cv.getTickFrequency()
    print('Time consume: %f s' % time)

    cv.waitKey(0)  # 程序等待指定的时间后执行以下语句(时间单位为：ms)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
