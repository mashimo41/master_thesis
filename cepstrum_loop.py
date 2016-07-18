#coding:utf-8
import cv2
import sys
import numpy as np
import scipy.fftpack
import scipy.signal
import scipy.ndimage
import makeBlur_bkp as blur
import findMin as find

'''
@file    cepstrum_loop.py
@author  taiki mashimo <mashimo.taiki@ac.jaxa.jp>
@date    2016-7-11
@brief   calculate blur (length,theta) from blur image sequence for 0609-experiment
'''

'''
Usage:
python cepstrum_loop.py > result.csv
Input  :blur images( sequence of .png files)
Output :blur length and orientation( csv file)
you can open csv file in libre officeCalc!
'''

def calcBlurparameter(imB):

    ######################################################################
    # 前処理
    ######################################################################

    # Smooting
    imB = cv2.GaussianBlur(imB, (11, 11), 0)

    # 微分
    imB = cv2.Laplacian(imB, cv2.CV_64F, 3)

    ######################################################################
    # ケプストラム計算
    ######################################################################

    fft = scipy.fftpack.fft2(imB)
    mag = np.abs(fft)
    log = 2 * np.log(mag)
    log = scipy.ndimage.fourier_gaussian(log, 2.) # LPF
    cpB = scipy.fftpack.ifft2(log)
    cpB = scipy.fftpack.fftshift(cpB)
    cpB = cpB.astype(np.float)

    ######################################################################
    # FInd min
    ######################################################################

    # find 
    minId = find.findMin(cpB)
    minId_sub =find.LeastSquare(cpB, minId)

    #ぶれ幅L 
    L = np.sqrt((minId_sub[0]-cpB.shape[0]/2)**2+(minId_sub[1]-cpB.shape[1]/2)**2)

    # 方向th
    dx = minId_sub[0]-cpB.shape[0]/2
    dy = minId_sub[1]-cpB.shape[1]/2
    th = -np.arctan(dy/dx)*(180.0/np.pi)

    return L, th
    

if __name__ == "__main__":

    start = 10
    end = 76
    file_number = np.arange(start, end+1, 1)

    for i in file_number:

        # filename = "./0609Chofu_experiment/images/fc2_save_2016-06-09-161411-0%03d.png" % i
        filename = "./0609Chofu_experiment/output/fc2_save_2016-06-09-161411-0%03d_undistorted.png" % i
        imB = cv2.imread(filename,0).astype(np.float)
    
        L, th = calcBlurparameter(imB)

        print i, ",",L, ",", th
