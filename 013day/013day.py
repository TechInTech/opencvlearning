# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ********************* 超大图像二值化及其应用 **********************
"""
1、分块
2、全局阈值，局部阈值

"""

def big_image(image):
    print(image.shape)
    cw, ch = 256, 256
    h, w = image.shape[:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            roi = gray[row : row + ch, col : col + cw]

            # ************************************
            # method 1
            # ret, dst = cv.threshold(roi, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) # 全局阈值

            # method 2
            # dst = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 127, 20) # 局部阈值

            # gray[row : row + ch, col : col + cw] = dst
            # print(np.std(dst), np.mean(dst))
            # ************************************

            # method 3 (全局阈值和图像ROI与空白图像过滤)
            print(np.std(roi), np.mean(roi))
            dev = np.std(roi)
            if dev < 15:
                gray[row : row + ch, col : col + cw] = 255
            else:
                ret, dst = cv.threshold(roi, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
                gray[row : row + ch, col : col + cw] = dst

    cv.imwrite('../picsrc/result_binary.png', gray)


def main():
    src = cv.imread('../picsrc/bigpic.jpg')
    cv.namedWindow('pic', cv.WINDOW_AUTOSIZE)
    cv.imshow('pic', src)

    t1 = cv.getTickCount()

    big_image(src)

    t2 = cv.getTickCount()
    time = (t2 - t1)/cv.getTickFrequency()
    print('Time consume: %f s' % time)

    cv.waitKey(0)  # 程序等待指定的时间后执行以下语句(时间单位为：ms)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
