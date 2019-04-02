# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np

def extrace_object():
    # 打开摄像头
    capture = cv.VideoCapture(0)
    while(True):
        ret, frame = capture.read()  # 若能读取数据ret返回True, 否则返回Falseframe 为实时图片
        # frame = cv.flip(frame, 1)
        if ret == False:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # 将色彩空间由RGB转化为HSV空间
        #绿色通道的范围（HSV色彩空间）
        lower_hsv = np.array([37, 43, 46])   # 分别绿色通道中H、S、V的最小值
        upper_hsv = np.array([77, 255, 255]) # 分别绿色通道中H、S、V的最大值

        # # *色通道的范围（HSV色彩空间）
        # lower_hsv = np.array([100, 43, 46])
        # upper_hsv = np.array([124, 255, 255])

        mask = cv.inRange(frame, lowerb = lower_hsv, upperb = upper_hsv) # 通过界定HSV空间图像中色彩颜色的通道，筛选(跟踪)出所需要的颜色
        dst = cv.bitwise_and(frame, frame, mask = mask) # 通过逻辑运算显示所要筛选的对象颜色
        cv.imshow('video', frame)     # 显示原图后的video
        # cv.imshow('testVideo', mask)  # 显示处理后的video
        cv.imshow('testVideo', dst)  # 只显示处理后的对象
        c = cv.waitKey(40)
        if c == 27:
            break

# src = cv.imread('musictreecut.jpg')
# cv.namedWindow('music', cv.WINDOW_AUTOSIZE)
# cv.imshow('music', src)
t1 = cv.getTickCount()

extrace_object() # 抽取图像\视频中的指定对象(颜色)

# RGB图像的分离
# b, g, r = cv.split(src)
# cv.imshow('blue', b)
# cv.imshow('green', g)
# cv.imshow('red', r)
#
# # R、G、B合成
# src = cv.merge([b, g, r])
# src[:, :, 0] = 0
# cv.imshow('merged image', src)
# cv.imwrite('./yellowtree.png', src)

t2 = cv.getTickCount()
time = (t2 - t1)/cv.getTickFrequency()

print('Time: %f ms' % time)
cv.waitKey(0)

cv.destroyAllWindows()
