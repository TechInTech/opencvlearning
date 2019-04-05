# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np

# ********************* 对象测量及其应用 **********************
"""
1、弧长与面积
   > 轮廓发现
   > 计算每个轮廓的弧长与面积，像素面积
2、多边形拟合
3、几何矩计算
4、API介绍
"""


def measure_object(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)   # 灰度处理
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    print('threshold value: %s' % ret)
    cv.imshow('binary image', binary)
    dst = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    cotImage, contours, hireachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)  # 计算轮廓面积
        x, y, w, h = cv.boundingRect(contour)  # 外接矩形坐标
        rate = min(w, h) / max(w, h)   # 宽高比
        print('rectangle rate: %s' % rate)
        mm = cv.moments(contour)   # 几何矩
        print(type(mm))
        cx = mm['m10'] / mm['m00']   # 几何中心坐标 cx、cy
        cy = mm['m01'] / mm['m00']   #
        cv.circle(dst, (np.int(cx), np.int(cy)), 3, (0, 255, 255), -1)  # 由中心点绘制几何外接轮廓
        # cv.rectangle(dst, (x, y), (x + w, y + h), (0, 0, 255), 2)    # 对每个轮廓绘制一个外接矩形
        print('contour area: %s' % area)
        approxCurve = cv.approxPolyDP(contour, 2, True)   # 多边形逼近
        print(approxCurve.shape)
        if approxCurve.shape[0] == 4:
            cv.drawContours(dst, contours, i, (0, 255, 0), 2)
        if approxCurve.shape[0] == 5:
            cv.drawContours(dst, contours, i, (255, 0, 0), 2)
        if approxCurve.shape[0] == 6:
            cv.drawContours(dst, contours, i, (0, 0, 255), 2)
        if approxCurve.shape[0] == 8:
            cv.drawContours(dst, contours, i, (255, 0, 255), 2)
        if approxCurve.shape[0] == 9:
            cv.drawContours(dst, contours, i, (150, 155, 0), 2)
        if approxCurve.shape[0] == 14:
            cv.drawContours(dst, contours, i, (9, 155, 150), 2)
        if approxCurve.shape[0] == 16:
            cv.drawContours(dst, contours, i, (0, 200, 250), 2)
        if approxCurve.shape[0] == 12:
            cv.drawContours(dst, contours, i, (250, 100, 0), 2)
    cv.imshow('measure contours', dst)


def main():
    src = cv.imread('../picsrc/jiji.jpg')
    cv.namedWindow('pic', cv.WINDOW_AUTOSIZE)
    cv.imshow('template', src)

    t1 = cv.getTickCount()

    # 1
    measure_object(src)

    t2 = cv.getTickCount()
    time = (t2 - t1)/cv.getTickFrequency()
    print('Time consume: %f s' % time)

    cv.waitKey(0)  # 程序等待指定的时间后执行以下语句(时间单位为：ms)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
