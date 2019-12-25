# -*- coding: utf-8 -*-
# Time    : 2019/4/15 16:40
# Author  : 辛放
# Email   : 1374520110@qq.com
# ID      : SZ160110115
# File    : main.py
# Software: PyCharm

import numpy as np
import cv2
def process(img):
    a = 0.1
    k = 0
    while k < 8:
        for i in range(0,img.shape[0] - 1):
            for j in range(0,img.shape[1] - 1):
                Uxx0 = float(img[i+1,j,0] - img[i,j,0] * 2 + img[i-1,j,0])
                Uxx1 = float(img[i+1,j,1] - img[i,j,1] * 2 + img[i-1,j,1])
                Uxx2 = float(img[i+1,j,2] - img[i,j,2] * 2 + img[i-1,j,2])
                Uyy0 = float(img[i,j+1,0] - img[i,j,0] * 2 + img[i,j-1,0])
                Uyy1 = float(img[i,j+1,1] - img[i,j,1] * 2 + img[i,j-1,1])
                Uyy2 = float(img[i,j+1,2] - img[i,j,2] * 2 + img[i,j-1,2])
                img[i,j,0] = img[i,j,0] + a*(Uxx0 + Uyy0)
                img[i,j,0] = img[i,j,0] + a*(Uxx1 + Uyy1)
                img[i,j,0] = img[i,j,0] + a*(Uxx2 + Uyy2)
            k = k+1

def main():

    #读取图像
    img_5 = cv2.imread('5.bmp')
    #均值模糊处理去噪音
    blured = cv2.blur(img_5,(5,5))
    cv2.imwrite('5_b.bmp',blured)
    #中值模糊处理去噪音
    mblured = cv2.medianBlur(img_5,5)
    cv2.imwrite('5_mb.bmp',mblured)
    #高斯模糊处理去噪音
    process(img_5)
    cv2.imwrite('5_p.bmp',img_5)

    #根据均值生成二值化图片
    gray = cv2.cvtColor(blured,cv2.COLOR_RGB2GRAY)
    h,w = gray.shape[:2]
    m = np.reshape(gray,[1,w*h])
    mean = m.sum()/(w*h)
    ret,gray_b = cv2.threshold(gray,mean,255,cv2.THRESH_BINARY)
    cv2.imwrite('5_b_g.bmp',gray_b)

    gray = cv2.cvtColor(mblured,cv2.COLOR_RGB2GRAY)
    h,w = gray.shape[:2]
    m = np.reshape(gray,[1,w*h])
    mean = m.sum()/(w*h)
    ret,gray_b = cv2.threshold(gray,mean,255,cv2.THRESH_BINARY)
    cv2.imwrite('5_mb_g.bmp',gray_b)



    img_15 = cv2.imread('15.jpg')
    blured = cv2.blur(img_15,(5,5))
    cv2.imwrite('15_b.bmp',blured)
    mblured = cv2.medianBlur(img_15,5)
    cv2.imwrite('15_mb.bmp',mblured)
    process(img_15)
    cv2.imwrite('15_p.bmp',img_15)

    gray = cv2.cvtColor(blured,cv2.COLOR_RGB2GRAY)
    h,w = gray.shape[:2]
    m = np.reshape(gray,[1,w*h])
    mean = m.sum()/(w*h)
    ret,gray_b = cv2.threshold(gray,mean,255,cv2.THRESH_BINARY)
    cv2.imwrite('15_b_g.bmp',gray_b)

    gray = cv2.cvtColor(mblured,cv2.COLOR_RGB2GRAY)
    h,w = gray.shape[:2]
    m = np.reshape(gray,[1,w*h])
    mean = m.sum()/(w*h)
    ret,gray_b = cv2.threshold(gray,mean,255,cv2.THRESH_BINARY)
    cv2.imwrite('15_mb_g.bmp',gray_b)


    img_25 = cv2.imread('25.bmp')
    blured = cv2.blur(img_25,(5,5))
    cv2.imwrite('25_b.bmp',blured)
    mblured = cv2.medianBlur(img_25,5)
    cv2.imwrite('25_mb.bmp',mblured)

    gray = cv2.cvtColor(blured,cv2.COLOR_RGB2GRAY)
    h,w = gray.shape[:2]
    m = np.reshape(gray,[1,w*h])
    mean = m.sum()/(w*h)
    ret,gray_b = cv2.threshold(gray,mean,255,cv2.THRESH_BINARY)
    cv2.imwrite('25_b_g.bmp',gray_b)

    gray = cv2.cvtColor(mblured,cv2.COLOR_RGB2GRAY)
    h,w = gray.shape[:2]
    m = np.reshape(gray,[1,w*h])
    mean = m.sum()/(w*h)
    ret,gray_b = cv2.threshold(gray,mean,255,cv2.THRESH_BINARY)
    cv2.imwrite('25_mb_g.bmp',gray_b)

if __name__ == '__main__':
    main()


