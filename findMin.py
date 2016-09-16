#coding:utf-8
import numpy as np
import cv2
import sys
import scipy.optimize
import matplotlib.pyplot as plt

def findMin(cpB, roi_size=5, debug=False):
    
    # Extract index of row's local minimum
    cpB_row = cpB[:, cpB.shape[1]/2]
    minId_row = cpB_row.argmin()

    # Set ROI  around local minimum by roi_size
    r = roi_size/2
    roi = cpB[minId_row-r:minId_row+r+1, cpB.shape[1]/2-r:cpB.shape[1]/2+r+1]
    if debug==True:
        print 'ROI:', roi

    # Extract index of roi's local minimum
    minvId_roi = roi.argmin() + 1
    minId_roi = (minvId_roi / (2*r+1) + 1), (minvId_roi % (2*r+1))

    # Calculate index of cpB's local minimum
    minId = ((minId_roi[0]-3) + minId_row, (minId_roi[0]-3) + cpB.shape[1]/2)

    return minId

def f(x,a,b,c):
    return a*x**2 + b*x + c

def LeastSquare(cpB, minId, roi_size=5, debug=False):

    # Range of Interest
    r = roi_size/2

    # Set independent variable
    id_col = np.arange(minId[1]-r, minId[1]+r+1, 1.)
    id_row = np.arange(minId[0]-r, minId[0]+r+1, 1.)

    # Set dependent variable
    roi_col = cpB[minId[0],minId[1]-r:minId[1]+r+1]
    roi_row = cpB[minId[0]-r:minId[0]+r+1,minId[1]]

    # Least Square Method:2nd order function fitting
    para_col, val_col= scipy.optimize.curve_fit(f, id_col, roi_col)
    para_row, val_row= scipy.optimize.curve_fit(f, id_row, roi_row)

    # calculate  of 2nd order function
    # y = p(x-a)^2 + b
    a_col =  -(para_col[1]/(2*para_col[0]))
    b_col =  -para_col[1]**2/(4*para_col[0])+para_col[2]
    a_row =  -(para_row[1]/(2*para_row[0]))
    b_row =  -para_row[1]**2/(4*para_row[0])+para_row[2]

    if debug==True:
        print 'cols (a,b):', a_col, b_col
        print 'rows (a,b):', a_row, b_row

    # Minimum coordinate:(a_row, a_col), Minimum value: mean(b_row, b_col)
    return (a_row, a_col), (b_row + b_col) / 2


if __name__ == "__main__":

    cpB = cv2.imread(sys.argv[1], 0)

    minId = findMin(cpB, debug=True)
    print minId
    (a, b), c = LeastSquare(cpB, minId)
    
    print a, b, c
