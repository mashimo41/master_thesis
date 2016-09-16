#coding:utf-8
import numpy as np
import cv2
import sys
import scipy.fftpack
import skimage.draw

'''
@file    slidingWindow.py
@author  taiki mashimo <mashimo.taiki@ac.jaxa.jp>
@date    2016-9-14
@brief   sliding window sample
'''

if __name__ == "__main__":

    # Load gray scale image as float
    img = cv2.imread(sys.argv[1],0).astype(np.float)
    # img = cv2.resize(img, (512, 512))
    img = cv2.resize(img, (256, 256))
    
    # Set window size
    kernelSize = (20,20)

    # Set step size
    # step = (20, 20)
    step = (1, 1)

    # Kinematic parameters(pixel)
    Rx = 0.0*(np.pi/180)
    Ry = 0.0*(np.pi/180)
    Rz = 0.0*(np.pi/180)
    Tx = 0.0
    Ty = 0.0
    Tz = 20.0
    Z = 1000.0
    f = 1000.0

    # Create array for visualizing
    vis = np.zeros((img.shape[0]*100, img.shape[1]*100, 3), np.uint8)

    # Create array for kernel
    h = kernelSize[0]
    w = kernelSize[1]
    # kernel = np.zeros((h, w, 3), np.uint8)

    # Create array for output
    imB = np.zeros(img.shape, np.float)

    # Sliding window
    for i in xrange(0+h/2, img.shape[0]-h/2+1, step[0]):
        for j in xrange(0+w/2, img.shape[1]-w/2+1, step[1]):
            kernel = np.zeros(kernelSize, np.float)
            kernel = cv2.resize(kernel, (kernelSize[0]*100, kernelSize[1]*100))

            # Calculate flow vector at current pixel
            x = (float)(j - img.shape[1]/2)
            y = (float)(i - img.shape[0]/2)
            dx = (x*y/f)*Rx + ((f**2+x**2)/f)*Ry - y*Rz -(f/Z)*Tx + (x/Z)*Tz
            dy = ((f**2+y**2)/f)*Rx + (x*y/f)*Ry + x*Rz -(f/Z)*Ty + (y/Z)*Tz

            dx = (int)(100 * dx)
            dy = (int)(100 * dy)
            print dx, dy

            # Create kernel
            k_x = kernel.shape[1]/2
            k_y = kernel.shape[0]/2
            # cv2.line(kernel, (k_x-dx/2, k_y-dy/2), (k_x+dx/2, k_y+dy/2), (255,255,255), 1)#, cv2.CV_AA)
            # # print kernel.shape
            # kernel = cv2.cvtColor(kernel, cv2.COLOR_BGR2GRAY)
            # kernel = kernel.astype(np.float)/np.amax(kernel)

            rr, cc, val = skimage.draw.line_aa(k_y-dy/2, k_x-dx/2, k_y+dy/2, k_x+dx/2)
            kernel[rr, cc] = val * 1.0
            cv2.line(vis, (j*100-dx/2, i*100-dy/2), (j*100+dx/2, i*100+dy/2), (255,255,255), 1)
            # scikit-image, PIL

            # Convolution
            roi = img[i - h/2:i + h/2, j - w/2:j + w/2]
            roi = cv2.resize(roi, (kernelSize[0]*100, kernelSize[1]*100))
            roi = roi * kernel
            roi = cv2.resize(roi, (kernelSize[0], kernelSize[1]))
            # print np.amax(roi)

            imB[i, j] = np.sum(roi)#/10

            # Show window contents
            cv2.namedWindow("roi", cv2.WINDOW_NORMAL)
            cv2.namedWindow("kernel", cv2.WINDOW_NORMAL)
            cv2.namedWindow("optical flow", cv2.WINDOW_NORMAL)
            cv2.imshow("roi", roi/np.amax(roi))
            cv2.imshow("kernel", kernel)

            cv2.waitKey(1)

    # cv2.imwrite("sim_imB.jpg", 255*imB/np.amax(imB))
    vis = cv2.resize(vis, (2048, 2048))
    cv2.imshow("optical flow", vis)

    a = 100
    img = imB#[img.shape[0]/2-a:img.shape[0]/2+a, img.shape[1]/2-a:img.shape[1]/2+a]

    # img = imB[20:50, 20:50]

    fft = scipy.fftpack.fft2(img)
    # H = scipy.fftpack.fftshift(H)
    mag = np.abs(fft)
    log = np.log(mag**2)
    cpB = scipy.fftpack.ifft2(log)
    cpB = scipy.fftpack.fftshift(cpB)
    cpB = cpB.astype(np.float)

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img/np.amax(img))
    cv2.namedWindow("mag", cv2.WINDOW_NORMAL)
    cv2.imshow("mag", log/np.amax(log))
    cv2.namedWindow("cpB", cv2.WINDOW_NORMAL)
    cv2.imshow("cpB", 10*cpB/np.amax(cpB)+0.3)
    cv2.waitKey(-1)
