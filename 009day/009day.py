# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ********************* 图像直方图及其应用 **********************
"""
Bin的大小 = 图像中不同像素值的个数 / Bin的数目

例如：
    对14位的图像，创建256 个bin直方图

    BIn size = 2^14 / 256 = 64

应用：
1、直方图均衡化：

    直方图均衡化的作用是图像增强。这种方法对于背景和前景都太亮或者太暗的图像非常有用

    第一个问题。均衡化过程中，必须要保证两个条件：①像素无论怎么映射，一定要保证原来的大小关系不变，较亮的区域，依旧是较亮的，较暗依旧暗，只是对比度增大，绝对不能明暗颠倒；②如果是八位图像，那么像素映射函数的值域应在0和255之间的，不能越界。综合以上两个条件，累积分布函数是个好的选择，因为累积分布函数是单调增函数（控制大小关系），并且值域是0到1（控制越界问题），所以直方图均衡化中使用的是累积分布函数。

    第二个问题。累积分布函数具有一些好的性质，那么如何运用累积分布函数使得直方图均衡化？比较概率分布函数和累积分布函数，前者的二维图像是参差不齐的，后者是单调递增的。

    -->全局均衡化：
    -->局部均衡化：

2、直方图比较
    比较依据：
    巴氏距离： 巴氏距离越大，说明越不相似；越小越相似，值完全匹配为0，完全不匹配则为1
    相关性： 越小，越不相似；越大相关性越大，越相似
            (相关性越强，相关系数就会越接近±1，相关性越弱，相关系数越接近0)
    卡方： 越大，说明两张图相差越大
            (卡方比较和相关性比较恰恰相反，相关性比较的值为0，相似度最低，越趋近于1，相似度越低；卡方比较则是，值为0时说明H1= H2，这个时候相似度最高。卡方比较来源于卡方检验，卡方检验就是统计样本的实际观测值与理论推断值之间的偏离程度，实际观测值与理论推断值之间的偏离程度就决定卡方值的大小，卡方值越大，越不符合；卡方值越小，偏差越小，越趋于符合，若两个值完全相等时，卡方值就为0，表明理论值完全符合)
    十字交叉：
"""

# 绘制图像直方图
# cv.calcHist 参数解释   参考网络博文：https://blog.csdn.net/YZXnuaa/article/details/79231817

def image_hist(image):
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color = color, label= color[0])
        plt.xlim([0, 256])
    plt.legend(loc = 'upper right')
    plt.show()


# 全局均衡化
def equalHist(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    dst = cv.equalizeHist(gray)
    cv.imshow('equalHist', dst)


# 局部均衡化
def clahe_Hist(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    dst = clahe.apply(gray)
    cv.imshow('clahe', dst)


# 得到直方图
def create_rgb_hist(image):
    h, w, c = image.shape
    rgbHist = np.zeros([16 * 16 * 16, 1], np.float32)
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int(b / bsize) * 16 * 16 + np.int(g / bsize) * 16 + np.int(r / bsize)
            rgbHist[np.int(index), 0] = rgbHist[np.int(index), 0] + 1
    return rgbHist


# 直方图比较
def hist_compare(image1, image2):
    hist1 = create_rgb_hist(image1)
    hist2 = create_rgb_hist(image2)
    plt.plot(hist1)
    plt.plot(hist2)
    match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    match4 = cv.compareHist(hist1, hist2, cv.HISTCMP_INTERSECT)
    print("巴氏距离:{}, 相关性:{}, 卡方:{}, 十字交叉性:{}".format(match1, match2, match3, match4))
    plt.show()


def main():
    # src = cv.imread('../picsrc/cai.jpg')
    src = cv.imread('../picsrc/darkimage.png')
    print(src.shape)
    cv.namedWindow('People', cv.WINDOW_AUTOSIZE)
    cv.imshow('cai', src)

    t1 = cv.getTickCount()

    # 显示直方图
    # image_hist(src)

    #全局均衡化
    # equalHist(src)

    # 局部均衡化
    # clahe_Hist(src)

    # 直方图比较
    src1 = cv.imread('../picsrc/musictreecut.jpg')
    print(src1.shape)
    cv.imshow('tree', src1)

    hist_compare(src, src1)




    t2 = cv.getTickCount()
    time = (t2 - t1)/cv.getTickFrequency()
    print('Time consume: %f s' % time)


    cv.waitKey(0)  # 程序等待指定的时间后执行以下语句(时间单位为：ms)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
