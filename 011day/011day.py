# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ********************* 模板匹配及其应用 **********************
"""

"""


def template(tpl, target):
    cv.imshow('template', tpl)
    cv.imshow('target', target)
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    th, tw = tpl.shape[:2]
    for md in methods:
        print(md)
        result = cv.matchTemplate(target, tpl, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        if md == cv.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0] + tw, tl[1] + th)
        cv.rectangle(target, tl, br, (0, 0, 255), 2)
        # cv.imshow('match-'+np.str(md), target)
        cv.imshow('match-'+np.str(md), result)


def main():
    src = cv.imread('../picsrc/pickcai.jpg')
    print(src.shape)
    cv.namedWindow('pic', cv.WINDOW_AUTOSIZE)
    # cv.imshow('cai', src)

    t1 = cv.getTickCount()

    src1 = cv.imread('../picsrc/cai.jpg')
    # print(src1.shape)

    template(src, src1)

    t2 = cv.getTickCount()
    time = (t2 - t1)/cv.getTickFrequency()
    print('Time consume: %f s' % time)


    cv.waitKey(0)  # 程序等待指定的时间后执行以下语句(时间单位为：ms)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
