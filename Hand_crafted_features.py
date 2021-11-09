
import cv2
import numpy as np
from skimage.feature import greycomatrix, local_binary_pattern


# HSV   
def HSV(input):
    img_bgr =np.array([input[:,:,2],input[:,:,1],input[:,:,0]])
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return img_hsv

def LAB(input):
    img_bgr =np.array([input[:,:,2],input[:,:,1],input[:,:,0]])      
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return img_lab
#Define a function to find the third-order color moments
'''def var(x = None):
    mid = bp.mean(((x-x.mean())**3))
    return np.sign(mid)*abs(mid)**(1/3)'''
#sobeldetection
def SOBEL(input):
    img_bgr =np.array([input[:,:,2],input[:,:,1],input[:,:,0]])       
    img_gray =cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    #img_blur = cv2.GaussianBlur(img_gray,(3,3),0)
    #img_rgb =np.array(input[:,:,0],input[:,:,1],input[:,:,2])
    sobelx = cv2.Sobel(img_gray, -1, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, -1, 0, 1, ksize=3)
    sobelxy= cv2.Sobel(img_gray, -1, 1, 1, ksize=3)
    sobelall = np.array([sobelx,sobely,sobelxy])
    return sobelall

def PREWITT(input):
    img_bgr =np.array([input[:,:,2],input[:,:,1],input[:,:,0]])       
    img_gray =cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype = int)
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype = int)
    x = cv2.filter2D(img_gray, cv2.CV_165, kernelx)
    y = cv2.filter2D(img_gray, cv2.CV_165, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return Prewitt

#local binary pattern
def LBP(input):
    radius = 1
    n_points = 8
    #image = np.array(input[:,:,0],input[:,:,1],input[:,:,2])
    img_bgr =np.array([input[:,:,2],input[:,:,1],input[:,:,0]])       
    img_gray =cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    #lbp = local_binary_pattern(img_gray,n_points, radius)
    lbp = np.copy(input)
    for channel in (0,1,2,3):
        lbp[:,:,channel] = local_binary_pattern(input[:,:,channel],n_points,radius)
    return lbp

def getNDVI(data_win):
    R = data_win[:, :, 0]
    NIR = data_win[:, :, 3]
    ndvi = (NIR - R) / (NIR + R)
    return ndvi

def getMSAVI(data_win, L=1):  # for sparse vegetation area
    R = data_win[:, :, 0]
    NIR = data_win[:, :, 3]
    # savi = ((NIR - R) / (NIR + R + L)) * (1 + L)
    msavi = (2 * NIR + 1 - np.sqrt((2 * NIR + 1)**2 - 8 * (NIR - R))) / 2
    return msavi

def getVARI(data_win):
    R = data_win[:, :, 0]
    G = data_win[:, :, 1]
    B = data_win[:, :, 2]
    vari = (G - R) / (G + R - B)
    return vari

def getARVI(data_win):
    R = data_win[:, :, 0]
    B = data_win[:, :, 2]
    NIR = data_win[:, :, 3]
    arvi = (NIR - (2 * R) + B) / (NIR + (2 * R) + B)
    return arvi

def getGCI(data_win):
    G = data_win[:, :, 1]
    NIR = data_win[:, :, 3]
    gci = NIR / G - 1
    return gci

def getSIPI(data_win):
    R = data_win[:, :, 0]
    B = data_win[:, :, 2]
    NIR = data_win[:, :, 3]
    sipi = (NIR - B) / (NIR - R)
    return sipi

def getGLCM(data_win):
    # imgBGR = data_win[:, :, 2:0]
    # imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
    imgGray = data_win # for test

    # calculate different glcm from different directions
    glcm_h = greycomatrix(imgGray, distances=[1], angles=[0], levels=256,
                        symmetric=False)
    glcm_v = greycomatrix(imgGray, distances=[1], angles=[np.pi / 2], levels=256,
                        symmetric=False)
    glcm_45 = greycomatrix(imgGray, distances=[1], angles=[np.pi / 4], levels=256,
                        symmetric=False)
    h = glcm_h[:, :, 0, 0]
    v = glcm_v[:, :, 0, 0]
    diag = glcm_45[:, :, 0, 0]
    joint = np.array([h, v, diag])  # return a 3*winsize*winsize matrix
    return joint