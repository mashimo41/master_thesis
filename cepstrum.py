#coding:utf-8
import scipy.fftpack
import scipy.signal
import scipy.ndimage
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import makeBlur_bkp as blur

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
    
    #グレースケールで画像読み込み．
    img = cv2.imread(sys.argv[1],0).astype(np.float)
    # cv2.imshow("input", img / np.amax(img))

    #PSF作成
    H = blur.makeBlurPSF(img.shape, L=20, th=np.pi*0.0)
    # H2 = blur.makeBlurPSF(img.shape, L=100, th=np.pi/3)
    # H = blur.makeDefocusPSF(img.shape, img.dtype, R=0.05)
    # H = H*H2
    #ぶれ画像読み込み
    # imB = cv2.resize(img,(img.shape[0]/4,img.shape[1]/4))
    # img = blur.mkBluredImg(img, H)
    # imb = img[100:600, 100:600] # test
    # imB = img[700:1200, 0:500] # chofu
    # imB = img[img.shape[0]/2-150:img.shape[0]/2+150, img.shape[1]/2-150:img.shape[1]/2+150] # for 0204 experiments
    # imB = img[img.shape[0]/2-256:img.shape[0]/2+256, img.shape[1]/2-256:img.shape[1]/2+256] # for lunar images
    # imB = img[img.shape[0]/2-512:img.shape[0]/2+512, img.shape[1]/2-512:img.shape[1]/2+512] # for lock images
    # imB = img[img.shape[0]/2-640:img.shape[0]/2+640, img.shape[1]/2-640:img.shape[1]/2+640] # for vertical blurred images

    #理論的には85X85以下だとぶれがでなくなる．
    a = 500
    # a = 150
    imB = img#[img.shape[0]/2-a:img.shape[0]/2+a, img.shape[1]/2-a:img.shape[1]/2+a]
    # imB = cv2.resize(imB, (400, 400), interpolation=cv2.INTER_CUBIC) #Bilinear interpolation 補間しなほうがぶれがよく出る
    # imB = img
    # for data of white board experiments
    # imB = img[100:300, 350:550] # theta = 45rad
    # imB = img[150:300, 150:300] # theta = 0rad
    # imB = img[100:400, 400:700] # theta = 0rad
    # imB = img[150:450, 200:500] # attitude, shift-variant blur

    plt.subplot(3, 1, 1)
    imB_plot =imB[:, imB.shape[1]/2]
    plt.plot(np.arange(imB.shape[0]), imB_plot)

    ######################################################################
    # 前処理
    ######################################################################

    # LPF bad influence, shorten blur length & noise-rich
    # imB = cv2.GaussianBlur(imB, (11, 11), 0)
    # plt.subplot(3, 1, 2)
    imB_plot_gaussian =imB[:, imB.shape[1]/2]
    plt.plot(np.arange(imB.shape[0]), imB_plot_gaussian)

    # Differentiate
    imB = cv2.Laplacian(imB, cv2.CV_64F, 7)
    # imB = cv2.GaussianBlur(imB, (9, 9), 0)
    # ret, imB = cv2.threshold(imB, 254.5, 20.0, cv2.THRESH_TOZERO)
    # ans1,ans2,ans2 = map(lambda x:x**2 if x>0 else 0,[a,b,c,d]) 
    # e = np.eye(3)
    # plt.imshow(e)
    # plt.draw()

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
    scale = 150.
    fft = scipy.fftpack.fft2(imB)
    fft = scipy.fftpack.fftshift(fft)
    mag = np.abs(fft)
    # mag = mag / np.amax(mag) * scale #シミュレーションの場合は外した方がいい
    #スペクトルの増幅（正規化してからの方がスケールを考えやすい）
    spB = 10*np.log(mag**2)

    # mask = np.ones(spB.shape, dtype=np.uint8)
    # ran = 130
    # mask[mask.shape[0]/2-ran:mask.shape[0]/2+ran, mask.shape[1]/2-ran:mask.shape[1]/2+ran] = 255
    # spT = cv2.bitwise_and(spB, spB, mask=mask)
    # print np.mean(spB[:50,:])
    # spT = np.where(mask>1, spB, 176)

    cv2.namedWindow("blured image spectrum", cv2.WINDOW_NORMAL)
    cv2.namedWindow("blured image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("blured image log spectrum", cv2.WINDOW_NORMAL)
    cv2.imshow("blured image", imB / np.amax(imB))
    cv2.imshow("blured image spectrum", mag/ np.amax(mag)+0.3)
    cv2.imshow("blured image log spectrum", spB/ np.amax(spB))
    # cv2.imshow("mask", mask)

    s0 = (img / np.amax(img))*255
    cv2.imwrite("img.jpg", s0)
    s = (imB / np.amax(imB))*255
    cv2.imwrite("imB.jpg", s)
    s2 = (mag / np.amax(mag)+0.3)*255
    cv2.imwrite("mag.jpg", s2)
    ######################################################################
    # ケプストラム計算・表示
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
    log = 2 * np.log(mag) #対数の方が負の値の変化がみやすい
    # mask = np.ones(log.shape, dtype=np.uint8)
    # ran = 100
    # mask[mask.shape[0]/2-ran:mask.shape[0]/2+ran, mask.shape[1]/2-ran:mask.shape[1]/2+ran] = 255
    # # spT = cv2.bitwise_and(spB, spB, mask=mask)
    # #print 'mean', np.sum(np.bitwise_and(np.real(log).astype(np.int), mask)) / np.count_nonzero(mask)
    # log = np.where(mask>1, log, 176.0)
    # ran = 50
    # log[log.shape[0]/2-ran:log.shape[0]/2+ran, log.shape[1]/2-ran:log.shape[1]/2+ran] = 0.0
    # print log
    # print "testtest"

    # LPF2
    log = scipy.ndimage.fourier_gaussian(log, 2.)
    log2 = scipy.fftpack.fftshift(log)
    cv2.imshow("test2", log2/np.amax(log2))

    cpB = scipy.fftpack.ifft2(log)
    cpB = scipy.fftpack.fftshift(cpB) #fftshift ifftshift
    cpB = cpB.astype(np.float)

    ###################################
    # ケプストラム表示
    cps_plot = cps[cpB.shape[0]/2, :]
    # plt.plot(np.arange(cps.shape[1]), cps_plot)

    cpH_plot = cpH[cpB.shape[0]/2, :]
    # plt.plot(np.arange(cps.shape[1]), cpH_plot)

    # cpB_plot = cpB[cpB.shape[0]/2, :]
    # plt.plot(np.arange(cpB.shape[1]), cpB_plot)
    cpB_plot =cpB[:, cpB.shape[1]/2]
    # cpB_plot = np.sum(cpB, axis=1)/cpB.shape[0]
    # cpB_plot[cpB.shape[1]/2-10:cpB.shape[1]/2+10] = 0 # for visibility
    plt.subplot(3, 1, 3)
    plt.ylim(-0.01, 0.01)
    plt.plot(np.arange(cpB.shape[0]), cpB_plot)
    # im = img[img.shape[0]/2+20, :]
    # plt.plot(np.arange(im.shape[0]), im)

    # imB_plot =imB[:, imB.shape[1]/2]
    # plt.plot(np.arange(imB.shape[0]), imB_plot)

    # 1/3s 20 pixel, 1/2s 41 pixel, 1/10s 6 pixel
    plt.draw()
    plt.pause(-1)

    cp_min = np.abs(np.amin(cpB)) if np.amin(cpB) < 0 else 0.0
    cv2.namedWindow("blured image cepstrum", cv2.WINDOW_NORMAL)
    cv2.namedWindow("blured image cepstrum2", cv2.WINDOW_NORMAL)
    cv2.imshow("blured image cepstrum", np.log((cpB + cp_min) / np.amax(cpB+cp_min) +0.3))
    # cv2.imshow("blured image cepstrum", (cpB + cp_min) / np.amax(cpB+cp_min)*100)

    # cv2.imshow("blured image cepstrum2", (cpB / np.amax(cpB))*10000.+0.3)
    # cv2.imshow("blured image cepstrum2", (cpB / np.amax(cpB))*10000.+0.5) #B
    # cv2.imshow("blured image cepstrum2", (cpB / np.amax(cpB))*1000.+0.5) #C
    cv2.imshow("blured image cepstrum2", (cpB / np.amax(cpB))*200+0.3) #simulate
    # cv2.imshow("blured image cepstrum2", cpB / np.amax(cpB))
    # cv2.imshow("blured image cepstrum2", (cpB / np.amax(cpB))*10000.+0.1) #垂直ブレ
    #ほとんど0の場合特徴が現れなくなる
    cv2.waitKey(0)

    cpB2 = (cpB / np.amax(cpB))*1000.+0.5
    # cpB2 = (cpB + cp_min) / np.amax(cpB+cp_min)-0.6
    cv2.imwrite("imB.jpg", (imB/np.amax(imB))*255)
    cv2.imwrite("spB.jpg", (spB/np.amax(spB))*255)
    cv2.imwrite("cpB.jpg", (cpB2)*255)
    

    #3D plot 重すぎ
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # u = np.arange(cpB_thres.shape[1])
    # v = np.arange(cpB_thres.shape[0])
    # uu, vv = np.meshgrid(u, v)
    # ax.scatter(uu, vv, cpB_thres, s=1)
    # plt.show(-1)

    #極小値の閾値を変えたい
    #ケプストラムを画像として表示する際のオフセットを変えたいGUI

    # raw_input()
