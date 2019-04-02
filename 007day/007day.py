# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np

# ********************* 高斯模糊 **********************
"""
1、
"""

def clamp(pv):
    # 保证图像各通道的值在0-255之间
    if pv > 255:
        return pv
    if pv < 0:
        return 0
    else:
        return pv


# 向图片中加入高斯噪声
def gaussian_noise(image):
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)  # 产生0-20之间的随机数

            # 获得各通道的值
            b = image[row, col, 0]  # blue
            g = image[row, col, 1]  # green
            r = image[row, col, 2]  # red

            # 向图片加入噪声
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    cv.imshow('gaussian image', image)


def main():
    src = cv.imread('../picsrc/cai.jpg')
    cv.namedWindow('People', cv.WINDOW_AUTOSIZE)
    cv.imshow('cai', src)

    t1 = cv.getTickCount()

    # 图片加噪声
    gaussian_noise(src)

    # 高斯模糊
    # dst = cv.GaussianBlur(src, (0, 0), 15)  # 高斯模糊
    dst = cv.GaussianBlur(src, (5, 5), 0)  # 高斯模糊应用：抑制高斯噪声
    cv.imshow('gaussian blur', dst)

    t2 = cv.getTickCount()
    time = (t2 - t1)/cv.getTickFrequency()
    print('Time consume: %f s' % time)


    cv.waitKey(0)  # 程序等待指定的时间后执行以下语句(时间单位为：ms)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
