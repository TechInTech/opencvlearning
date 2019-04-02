# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np

def access_pixels(image):
    # print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    # print('height: %d, width: %d, chanels: %d' % (height, width, channels))
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row, col, c]
                image[row, col, c] = 255 - pv
    cv.imshow('dark', image)
    cv.imwrite('./darkimage.png',image)

def inverse(image):
    det = cv.bitwise_not(image)
    cv.imshow('music', det)

def create_image():
    """
    img = np.zeros([400, 400, 3], np.uint8)
    img[:,:,0] = np.ones((400, 400)) * 255
    cv.imshow('music', img)


    img = np.zeros([400, 400, 1], np.uint8)
    img[:, :, 0] = np.ones((400, 400)) * 127
    cv.imshow('music', img)
    """

    img = np.zeros([400, 400, 3], np.uint8)
    for row in range(400):
        for col in range(400):
            for c in range(3):
                if row // 10 == 0:
                    img[row, col, c] = np.random.randint(0, 100)
                elif row // 8 == 0:
                    img[row, col, c] = np.random.randint(100, 150)
                elif row //6 == 0:
                    img[row, col, c] = np.random.randint(150, 200)
                elif row // 4 ==0:
                    img[row, col, c] = np.random.randint(200, 255)
                else:
                    img[row, col, c] = np.random.randint(0, 255)
    print(img)
    cv.imshow('music', img)

def color_space_transfrom(image):
    # 色彩空间之间的转化
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)
    cv.imwrite('./grayimage.png',gray)

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow('hsv', hsv)
    cv.imwrite('./hsvimage.png',hsv)

    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    cv.imshow('yuv', yuv)
    cv.imwrite('./yuvimage.png',yuv)

    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    cv.imshow('ycrcb', ycrcb)
    cv.imwrite('./yrcbimage.png',ycrcb)

    hls = cv.cvtColor(image, cv.COLOR_BGR2HLS)
    cv.imshow('hls', hls)
    cv.imwrite('./hlsimage.png',hls)


src = cv.imread('musictreecut.jpg')
cv.namedWindow('music', cv.WINDOW_AUTOSIZE)
# cv.imshow('music', src)
t1 = cv.getTickCount()
# access_pixels(src)
# create_image()
# inverse(src)
color_space_transfrom(src)
t2 = cv.getTickCount()
print('Time: %f ms' % ((t2 - t1)/cv.getTickFrequency()))
cv.waitKey(0)
cv.destroyAllWindows()
