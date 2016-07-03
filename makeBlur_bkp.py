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
フーリエ変換ライブラリ
#NumPy  provide DFT & FFT modlues
#SciPy  provide FFT only
#OpenCV provide DFT only
NumPyとSciPyでは，SciPyの方が若干高速で柔軟性があるためそちらを採用．
OpenCVは関数名はdftだが,FFTを実行している模様.しかも,OpenCVが最も高速.
だがFFTには最適なサイズ選択が必要で,OpenCVではこれを他の実装関数で
しなければならないのに対し,numpy,scipyでは自動でやってくれる.
python-OpenCVチュートリアルによるとOpenCVとnumpyではFFTの速度が4倍違う.

#FFTの関数は以下．fftの部分を
逆フーリエ変換ではifft，
離散フーリエ変換ではdft，
二次元フーリエ変換ではfft2
に置換する．
#np.fft.fft
#scipy.fftpack.fft
#cv2.dft
'''

'''
makeBlurPSF(make PSF for motion blur)
@param shape PSF shape(default: [512, 512])
@param dtype PSF data type(default: float)
@param L     blur length(default: 50 pixel)
@param th    motion orientation(default: PI/6 rad)
@param H     output PSF
'''
def makeBlurPSF(shape=[512, 512], dtype=float, L=50, th=np.pi/6):
    
    #PSFの初期化
    H = np.zeros(shape, dtype)
    
    #thetaが半時計回り正になるように変換
    th = -th

    #周波数軸の設定
    u = np.arange(-shape[1]/2, shape[1]/2, 1.0)
    #u軸(幅は周波数スペクトルと同じ)
    v = np.arange(-shape[0]/2, shape[0]/2, 1.0)
    #v軸(幅は周波数スペクトルと同じ)
    uu, vv = np.meshgrid(u, v) #u軸とv軸に基づいた格子点平面を作成
    
    #フィルタ行列　　並進ぶれモデル(motion blur)
    '''
    定義式： H(u,v) = sinc{ PI*z },z = L*( u*cos(theta) + v*sin(theta) )
    numpyの定義式: sinc(x) = sin(PI*x)/PI*x
    '''
    z = L*( uu*np.cos(th) + vv*np.sin(th)) / (shape[0]) #なぜ？
    H = np.sinc(z)

    return H
    
def makeDefocusPSF(shape=[512, 512], dtype=float, R=0.02):
    
    #PSFの初期化
    H = np.zeros(shape, dtype)
    
    #周波数軸の設定
    u = np.arange(-shape[1]/2, shape[1]/2, 1.0)
    v = np.arange(-shape[0]/2, shape[0]/2, 1.0)
    uu, vv = np.meshgrid(u, v)
    
    #フィルタ行列　　焦点ぼけモデル(out of focus, defocus)
    '''
    定義式： H(u,v) = 2 * J_1(z) / z, z = PI*R*sqrt(u^2+v^2)
    scipyの定義式: J_1(x) = ****** : 1st kind, 1st order Bessel function
    '''
    z = np.pi*R*np.sqrt(uu**2 + vv**2)
    H = scipy.special.jn(1,z)*(2/z)

    #原点のみJ_1->0, 1/z->infでJ_1/Zはnanが出力されてしまうので別定義
    H[shape[0]/2,shape[1]/2] = 1.0

    return H

def makeBlurCircleR(u=0.3, f=10.4, s=20, D=1.08):

    R = s * (D/2) * ((1/f) - (1/u) - (1/s))
    R = np.abs(R)
    return R

    '''
    ＊＊[tips3]二次元関数の行列格納＊＊
    二次元関数を画像との演算で用いる場合，2次元座標(x,y)と関数値f(x,y)を
    行列表現で離散的に表す必要がある．その際に，meshgridによる方法がmatlab的
    で一般に用いられる．
    例）
    x = np.arange(-50, 50, 1.0) #x軸
    y = np.arange(-50, 50, 1.0) #y軸
    xx, yy = np.meshgrid(x, y) #x軸とy軸に基づいた格子点平面を作成

    二次元sinc関数の実装(一応)
    r = np.sqrt(uu*uu+vv*vv)*0.1
    H = np.sinc(r)
    '''

'''
mkBluredImg(make blured image from original image)
@param src    Input image
@param H      Input Point Spread Function
@param imIFFT Output blured image
'''
def mkBluredImg(src, H):
    
    #1. FFT（高速フーリエ変換）
    fft = scipy.fftpack.fft2(src)
    '''
    ＊＊Fast Fourier Transformation usage＊＊
    フーリエ変換
    scipy.fftpack.fft(img)  :1 order FFT ex) signal processing
    scipy.fftpack.fft2(img) :2 order FFT ex) image processing
    
    逆フーリエ変換
    scipy.fftpack.ifft(img)  :1 order inverse-FFT ex) signal processing
    scipy.fftpack.ifft2(img)  :2 order inverse-FFT ex) image processing
    '''
    
    #2. スワップ
    fft = scipy.fftpack.fftshift(fft)
    '''
    単純にフーリエ変換を計算しただけだと，低周波領域が外側に分散してしまう
    ので，周波数スペクトルの点対称性を利用して，第一象限と第三象限，第二象
    限と第四象限を入れ替える．

    スワップしてからフィルタリング処理などをすることになる．
    また，逆フーリエ変換して画像を復元する際は再びスワップすることで象限の
    位置関係を戻してやる必要がある．
    '''
        
    '''フーリエ変換結果の正規化
    FFTの計算結果は複素数であるため，周波数スペクトルとして表示するため
    に，かつ，imshowで表示するために正規化する必要がある．
    (表示するための処理であって，フーリエ変換して周波数フィルタリングす
    るだけなら必要ない部分)

    処理手順
    1.画像のフーリエ変換のパワースペクトル(絶対値)を求める．
    (complex value: R+iX  ->  spectrum: sqrt(R^2+X^2) or |R+iX|)
    2.パワースペクトルを0〜1に正規化するため，logをとる．
    (logをとることで[0,inf)の定義域が[0,1)の範囲に収まる．(極限が1に収束する)
    真数条件から+1している)
    3.最大値で割り，255を掛けることで[0,255]の範囲に収める．
    (この操作により，操作2での最大値が255となる)
    4.cv2.imshowで表示するためにMatのデータ型であるuint8に型変換する．

    ※冒頭で述べたfloatで画像を扱う場合は，操作3の最大値で割る所まででよい．
    '''
        
    #3. フィルタリング:G=FH
    fft = fft*H

    #4. 逆フーリエ変換
    fft = scipy.fftpack.ifftshift(fft) #計算のため，swapを元に戻す
    ifft = scipy.fftpack.ifft2(fft) #IFFT
    #型変換で出力を無理やり実数(虚数切捨)にする．warningでるけど気にしない
    imIFFT  = ifft.astype(np.float)

    #5. 出力
    return imIFFT

def mkDefocusedImg(src):
    
    dst = np.zeros(src.shape, src.dtype)
    depth_map = np.ones(src.shape, src.dtype)*0.01
    depth_map[src.shape[0]/2-100:src.shape[0]/2+100, src.shape[1]/2-100:src.shape[1]/2+100] = 1.
    step = 32
    for y in range(src.shape[0]/step):
        for x in range(src.shape[1]/step):
            crop = src[y*step:y*step+step, x*step:x*step+step]
            depth = depth_map[y*step:y*step+step, x*step:x*step+step]
            R = depth.mean()
            # if R > 0.01: print R
            H = makeDefocusPSF(crop.shape, crop.dtype, R)
            # cv2.imshow("test", np.abs(H)/np.amax(np.abs(H)))
            blur = mkBluredImg(crop, H)
            dst[y*step:y*step+step, x*step:x*step+step] = blur

    return dst


    '''
    ifftの出力は複素数に設定されているが，
    出力が画像に戻るのであれば，値は実数(虚数0)になる気がするけど
    フィルタリングで伝達関数を掛けられているので複素数な気も...
    print ifftで見ると虚数成分も微小に存在するが，正しい結果な
    のか，計算誤差なのか...
    正しい結果の場合，絶対値をとるなど何らかの処理が必要
    点光源画像を見た感じ絶対値をとった方が誤差を生じている
    みたいだが，OpenCVのチュートリアルでは絶対値とってる．
    一般的には実数を取り出しているみたい．
    '''

if __name__ == "__main__":

    #グレースケールで画像読み込み．
    img = cv2.imread(sys.argv[1],0).astype(np.float)
    # cv2.imshow("input", img / np.amax(img))
    
    '''
    ＊＊float型で画像を扱ったほうが精度がよいらしい．＊＊
    uint8だと0-255までしか値を扱うことができず，
    FFTなどで少数点まで計算した値が，切り捨てられてしまう．
    また，分解能がuint8では256階調であるが，floatなら16bit＝2^16(6万)階調？
    より高精度な計算結果の格納が期待できる．
    ただ，cv2.imshowで表示するときは値を0-1に正規化しなければならない．
    正規化はimshow内で次のように行うとよい．
    cv2.imshow("img_name", img / np.amax(img) )
    
    ＊＊[tips2]入力画像は縦横比1:1の方がFFTが高速らしい．＊＊
    なので，入力画像は正方形のものとする．
    実際SLIMなど宇宙機の画像サイズも512×512程度．
    '''
    
    #テスト用理想点光源画像(低周波成分なし)
    psr = np.zeros((512,512),dtype=np.float)
    psr[400:410,200] = 255
    psr[100,200] = 255
    psr[255,200] = 255
    
    #PSF生成
    # R = makeBlurCircleR(f=50e-3, D=10e-3, u=0.06, s=1.)
    # H = makeDefocusPSF(img.shape, img.dtype, R=0.1)  
    # 焦点ボケのHは分解能の問題でabsの0点が浮いてしまう．
    H = makeBlurPSF(img.shape, img.dtype, L=0.05, th=np.pi*0.)

    #ぶれ画像生成
    imB = mkBluredImg(img, H)
    imC = mkBluredImg(psr, H)

    #地形依存のぼけ画像生成->ぶれさせる
    # imD = mkDefocusedImg(img)
    # H3 = makeBlurPSF(imD.shape, imD.dtype, 0.5, 0.0)
    # imE = mkBluredImg(imD, H3)

    #PSFの振幅スペクトル
    '''
    PSFは正規化されていることからlogを取ると負になるのでabsのみ表示
    '''
    spH = np.abs(H)

    #PSFの空間領域表現
    ifft = scipy.fftpack.ifft2(H)
    ifft = scipy.fftpack.fftshift(ifft)
    h = np.abs(ifft)
    #h = ifft.astype(np.float) #???

    #PSFのケプストラム
    logH = np.log((spH**2))
    logH2 = scipy.fftpack.fftshift(logH) #shift exchangable
    cpH = scipy.fftpack.ifft2(logH2)
    cpH = scipy.fftpack.fftshift(cpH)
    cpH = cpH.astype(np.float)

    #閾値処理：なぜか出てくるノイズを除去(周波数スペクトルの反射のためか)
    # cpH_thres = np.where(cpH<-0.7,cpH,0.0)
    cpH_thres = np.where(cpH<-0.5,cpH,0.0)

    #極小値抽出
    mixId = scipy.signal.argrelextrema(cpH_thres, np.less)
    print "This is ID:"
    print mixId

    #ぶれ幅、方向推定
    print np.sqrt((mixId[0]-256)**2+(mixId[1]-256)**2)
    print -np.arctan((mixId[0]-256.0)/(mixId[1]-256.0))*(180.0/np.pi)

    #描画
    #spH, cpH, logH
    logH_plot = logH[logH.shape[0]/2,:]
    plt.plot(np.arange(logH.shape[1]), logH_plot)
    plt.draw()
    plt.pause(0.1)

    cv2.imshow("blured image", imB / np.amax(imB))
    cv2.imshow("blured point optical source image", imC / np.amax(imC))
    cv2.imshow("h space domain", h / np.amax(h))
    cv2.imshow("H spectrum", spH / np.amax(spH))
    cv2.imshow("H cepstrum", cpH / np.amax(cpH) + 0.5)
    cv2.waitKey(0)

    # cv2.imwrite("h2.jpg", (h / np.amax(h))*255)
    # cv2.imwrite("spB2.jpg", (spB / np.amax(spB))*255)
    # cv2.imwrite("cpB2.jpg", (cpB / np.amax(cpB))*255)

