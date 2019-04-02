# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np


# 填充彩色图像
# 泛洪填充
def fill_color(image):
    copyImage = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)
    cv.floodFill(copyImage, mask, (30, 30), (0, 255, 255), (70, 70, 70), (50, 50, 50), cv.FLOODFILL_FIXED_RANGE)
    cv.imshow('fill_color', copyImage)

# floodFill(image, mask, seedPoint, newVal[, loDiff[, upDiff[, flags]]])

# 二值填充
def binary_fill_color(image):
    image = np.zeros([400, 400, 3], np.uint8)
    image[100:300, 100:300, :] = 255
    cv.imshow('binary_fill', image)

    mask = np.ones([402, 402, 1], np.uint8)
    mask[101:301, 101:301] = 0      # 只有mask为0的区域才能被填充
    cv.floodFill(image, mask, (200, 200), (100, 7, 255), cv.FLOODFILL_MASK_ONLY) # (200, 200)为选择点
    cv.imshow('binary_fill_color', image)


src = cv.imread('../picsrc/cai.jpg')
# src = cv.imread('../picsrc/musictreecut.jpg')
print(src.shape)
cv.namedWindow('People', cv.WINDOW_AUTOSIZE)
cv.imshow('cai', src)

# 泛洪填充
# fill_color(src)

# 二值填充
# binary_fill_color(src)



#"""
# *****获取ROI,以及对ROI区域进行处理
# face = src[20:330, 130:390]  # 通过指定图像的高、宽的取值范围，再通过numpy回去ROI
# gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
#
# # **灰度处理之后的结果不等于处理之前的的结果
# backface = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
# src[20:330, 130:390] = backface
#
# cv.imshow('face', src)
# cv.imshow('backface', backface)

### *****
other = src[380:530, 340:490]
cv.imshow('other', other)
cv.imwrite('../picsrc/pickcai.jpg', other)

cv.waitKey(0)

cv.destroyAllWindows()
