
import  cv2 as cv
import numpy as np

def video_capture():
    # 打开摄像头
    capture = cv.VideoCapture(0)
    while(True):
        ret, frame = capture.read()  # frame 为实时图片
        # print(ret)
        # print(frame)
        frame = cv.flip(frame, 1)
        cv.imshow('video', frame)
        c = cv.waitKey(50)
        if c == 27:
            break

def get_image_info(image):
    # 读取图像数据和属型
    print(type(image))   # 图像数据存储类型
    print(image.shape)   # 图像维度 height * width * channel
    print(image.size)    # 图像大小 像素数据
    print(image.dtype)   # 存储每个字节的位数
    pixel_data = np.array(image)  # 图像像素
    # print(pixel_data)


src = cv.imread('musictree.jpg')
cv.namedWindow('music', cv.WINDOW_AUTOSIZE)
cv.imshow('music', src)
get_image_info(src)
gray = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
cv.imwrite('gray.png', gray)
# video_capture()   #
cv.waitKey(0)

cv.destroyAllWindows()
