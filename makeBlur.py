#coding:utf-8
import scipy.fftpack
import scipy.special
import scipy.signal
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

'''
@file    makeBlur.py
@author  taiki mashimo <mashimo.taiki@ac.jaxa.jp>
@date    2016-1-22
@brief   make blured image from original image
'''


'''
makeBlurredImage(make PSF for motion blur)
@param shape PSF shape(default: [512, 512])
@param dtype PSF data type(default: float)
@param T     exposure time(default: 0.1)
@param V     motion velocity(default: 0.5)
@param th    motion direction(default: PI/6)
@param H     output PSF
'''
def makeBlurredImage(src, h):

    dst = np.zeros(src.shape, src.dtype)
    for y in xrange(src.shape[0]):
        for x in xrange(src.shape[1]):
            # window = src[y:y+h.shape[1], x:x+h.shape[0]]


'''
range(開始点,終了点,ステップサイズ)
ステップサイズは等差数列の公差
xrangeの方がメモリの使用が効率的
Sliding window: range(0, imageSize, 1)
Scan window   : range(0, imageSize, kernelSize)
Sliding windowではカーネルをすべての要素に畳み込む
Scan windowではカーネルサイズ毎に飛ばして畳み込む
'''

if __name__ == "__main__":

    #グレースケールで画像読み込み．
    img = cv2.imread(sys.argv[1],0).astype(np.float)
    cv2.imshow("input", img / np.amax(img))
    
    
    #テスト用理想点光源画像(低周波成分なし)
    psr = np.zeros((512,512),dtype=np.float)
    psr[400:410,200] = 255
    psr[100,200] = 255
    psr[255,200] = 255
    
