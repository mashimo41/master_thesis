#coding:utf-8
import scipy.fftpack
import scipy.signal
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import makeBlur_bkp as blur
import findMin as find

'''
@file    ****.py
@author  taiki mashimo <mashimo.taiki@ac.jaxa.jp>
@date    2016-1-22
@brief   make blured image from original image
'''

if __name__ == "__main__":


    ######################################################################
    # 画像の読み込み
    ######################################################################    

    # グレースケールで画像読み込み．
    img = cv2.imread(sys.argv[1],0).astype(np.float)

    # PSF作成
    H = blur.makeBlurPSF(img.shape, L=50, th=np.pi*0.0)

    # ぶれ画像読み込み
    # <シミュレーションの場合>
    # imB = blur.mkBluredImg(img, H)
    # <実画像実験の場合>
    a = 250
    imB = img#[img.shape[0]/2-a:img.shape[0]/2+a, img.shape[1]/2-a:img.shape[1]/2+a]

    ######################################################################
    # 前処理
    ######################################################################

    # 微分
    imB = cv2.Laplacian(imB, cv2.CV_64F, 3)

    ######################################################################
    # 空間周波数スペクトル計算・表示
    ######################################################################
    '''
    ＊＊＊周波数スペクトル＊＊＊
    FFTの結果F(x)=F[f(x)]を周波数スペクトルと呼ぶが,
    実際にはFFTの計算結果は複素数であるため,その振幅(magnitude)でスペクトル
    を表現する.特に強度(power，振幅の二乗)で表されることが多く,パワースペク
    トルと呼ばれる.
    また,人間の感覚は対数的であるとされているため,対数スペクトルの方がよい.
    さらに,工学ではこれをデジベルで表現する.
    振幅スペクトル                 |F(x)|
    パワースペクトル               |F(x)|^2
    対数振幅スペクトル             log|F(x)|
    対数パワースペクトル           log|F(x)|^2
    対数パワースペクトル(dB)       10log|F(x)|^2 = 20log|F(x)|
    画像として表示する場合,最後に0-255(floatなら0-1)に正規化する必要がある
    <周波数スペクトル表示までのフロー>
    img -> FFT -> shift -> abs() -> **2 -> 10log() -> /max() -> spectrum image
    '''
    scale = 100.
    fft = scipy.fftpack.fft2(imB)
    fft = scipy.fftpack.fftshift(fft)
    mag = np.abs(fft)
    # mag = mag / np.amax(mag) * scale #シミュレーションの場合は外した方がいい
    #スペクトルの増幅（正規化してからの方がスケールを考えやすい）
    spB = 10*np.log(mag**2)

    cv2.namedWindow("input", cv2.WINDOW_NORMAL)
    cv2.namedWindow("blured image spectrum", cv2.WINDOW_NORMAL)
    cv2.namedWindow("blured image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("blured image log spectrum", cv2.WINDOW_NORMAL)
    cv2.imshow("input", img / np.amax(img))
    cv2.imshow("blured image", imB / np.amax(imB))
    cv2.imshow("blured image spectrum", mag/ np.amax(mag)+0.3)
    cv2.imshow("blured image log spectrum", spB/ np.amax(spB))

    ######################################################################
    # ケプストラム計算
    ######################################################################

    '''
    ＊＊ケプストラム＊＊
    普通は実数領域を見るので，Re[C(x)]
    振幅ケプストラム   |C(x)|
    パワーケプストラム |C(x)|^2
    <ケプストラム計算のフロー>
    img -> FFT -> abs() -> **2 -> log() -> IFFT -> shift -> Re() -> cepstrum
    '''

    ###################################
    # 入力画像ケプストラム
    fft = scipy.fftpack.fft2(img)
    mag = np.abs(fft)
    log = np.log(mag**2)
    cps = scipy.fftpack.ifft2(log)
    cps = scipy.fftpack.fftshift(cps)
    cps = cps.astype(np.float)

    ###################################
    # PSFケプストラム
    H = scipy.fftpack.fftshift(H)
    mag = np.abs(H)
    log = np.log(mag**2)
    cpH = scipy.fftpack.ifft2(log)
    cpH = scipy.fftpack.fftshift(cpH)
    cpH = cpH.astype(np.float)

    ###################################
    # ぶれ画像ケプストラム
    fft = scipy.fftpack.fft2(imB)
    # ここでシフトしてはいけない．素のfftの絶対値をとる
    mag = np.abs(fft)
    # mag = mag / np.amax(mag) * scale
    #負の極小値が消えないように注意する．あと低周波が正になるようにする．
    log = 2 * np.log(mag)
    cpB = scipy.fftpack.ifft2(log)
    cpB = scipy.fftpack.fftshift(cpB)
    cpB = cpB.astype(np.float)

    ######################################################################
    # 閾値処理
    ######################################################################
    '''
    3項演算子where
    戻り値 = where(条件式,真での値,偽での値)
    '''
    # ノイズ除去
    cpB_thres = np.where(cpB<-0.005,cpB,0.0)
    # cpH_thres = np.where(cpH<-0.7,cpH,0.0)
    # cpB_thres = np.where(cpB<-0.4,cpB,0.0)

    # 低周波成分除去
    for j in xrange(0, cpB_thres.shape[1]):
        for i in xrange(0, cpB_thres.shape[0]):
            if ((i-cpB_thres.shape[0]/2)**2+(j-cpB_thres.shape[1]/2)**2) < 15**2:
                cpB_thres[i][j] = 0.0

    #極小値抽出
    mixId = scipy.signal.argrelextrema(cpB_thres, np.less)
    print "This is ID:"
    print mixId

    #ぶれ幅、方向推定 
    print np.sqrt((mixId[0]-imB.shape[0]/2)**2+(mixId[1]-imB.shape[1]/2)**2)
    dx = mixId[0]-imB.shape[0]/2
    dy = mixId[1]-imB.shape[1]/2
    # print dx, dy
    # print -np.arctan((mixId[0]-imB.shape[0]/2)/(mixId[1]-imB.shape[1]/2))*(180.0/np.pi)
    print -np.arctan(dy/dx)*(180.0/np.pi)


    ######################################################################
    # ケプストラム計算
    ######################################################################
    cps_plot = cps[cpB.shape[0]/2, :]
    # plt.plot(np.arange(cps.shape[1]), cps_plot)

    cpH_plot = cpH[cpB.shape[0]/2, :]
    # plt.plot(np.arange(cps.shape[1]), cpH_plot)

    # cpB_plot = cpB[cpB.shape[0]/2, :]
    # plt.plot(np.arange(cpB.shape[1]), cpB_plot)
    cpB_plot = cpB[:, cpB.shape[1]/2]
    plt.plot(np.arange(cpB.shape[0]), cpB_plot)

    plt.draw()
    plt.pause(0.1)

    cv2.namedWindow("blured image cepstrum", cv2.WINDOW_NORMAL)
    cv2.imshow("blured image cepstrum", (cpB / np.amax(cpB))*2000+0.3)
    cv2.waitKey(0)
    
