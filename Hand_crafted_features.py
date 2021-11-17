
import cv2
import numpy as np
from skimage.feature import greycomatrix, local_binary_pattern
from skimage import filters
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk


# HSV
def HSV(input):
    img_bgr =np.array([input[:,:,2],input[:,:,1],input[:,:,0]])
    img_bgr = img_bgr.transpose((1,2,0))
    img_hsv = cv2.cvtColor(np.float32(img_bgr), cv2.COLOR_BGR2HSV)
    return img_hsv

def LAB(input):
    img_bgr =np.array([input[:,:,2],input[:,:,1],input[:,:,0]])
    img_bgr = img_bgr.transpose((1,2,0))
    img_lab = cv2.cvtColor(np.float32(img_bgr), cv2.COLOR_BGR2LAB)
    return img_lab
#Define a function to find the third-order color moments
'''def var(x = None):
    mid = bp.mean(((x-x.mean())**3))
    return np.sign(mid)*abs(mid)**(1/3)'''
#sobeldetection
def SOBEL(input):
    img_bgr =np.array([input[:,:,2],input[:,:,1],input[:,:,0]])
    img_bgr = img_bgr.transpose((1,2,0))
    img_gray =cv2.cvtColor(np.float32(img_bgr), cv2.COLOR_BGR2GRAY)
    #img_blur = cv2.GaussianBlur(img_gray,(3,3),0)
    #img_rgb =np.array(input[:,:,0],input[:,:,1],input[:,:,2])
    sobelx = cv2.Sobel(img_gray, -1, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, -1, 0, 1, ksize=3)
    sobelxy= cv2.Sobel(img_gray, -1, 1, 1, ksize=3)
    sobelall = np.array([sobelx,sobely,sobelxy]).transpose((1, 2, 0))
    return sobelall

def PREWITT(input):
    img_bgr =np.array([input[:,:,2],input[:,:,1],input[:,:,0]])
    img_bgr = img_bgr.transpose((1,2,0))
    img_gray =cv2.cvtColor(np.float32(img_bgr), cv2.COLOR_BGR2GRAY)
    output = filters.prewitt(img_gray)
    return output[:,:,np.newaxis]

'''def PREWITT(input):
    img_bgr =np.array([input[:,:,2],input[:,:,1],input[:,:,0]])       
    img_bgr = img_bgr.transpose((1,2,0))
    img_gray =cv2.cvtColor(np.float32(img_bgr), cv2.COLOR_BGR2GRAY)
    #output = img_gray.copy()
    W, H = img_gray.shape
    lastrow = img_gray[-1,:]
    img_gray = np.vstack(img_gray,lastrow)
    img_gray = np.vstack(img_gray,lastrow)
    img_gray = np.vstack(img_gray,lastrow)
    lastcol = img_gray[:,-1]
    img_gray = np.vstack(img_gray,lastcol)
    img_gray = np.vstack(img_gray,lastcol)
    img_gray = np.vstack(img_gray,lastcol)
    new_image = np.zeros((W-3, H-3))
    new_imageX = np.zeros((W-3,H-3))
    new_imageY = np.zeros((W-3,H-3))
    s_suanziX = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])     
    s_suanziY = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])   
    for i in range(W-3):
        for j in range(H-3):
            new_imageX[i, j] = abs(np.sum(img_gray[i:i+3, j:j+3] * s_suanziX))
            new_imageY[i, j] = abs(np.sum(img_gray[i:i+3, j:j+3] * s_suanziY))
            new_image[i, j] = max(new_imageX[i,j] , new_imageY[i, j])
    return new_image
'''
#local binary pattern
def LBP(input):
    radius = 1
    n_points = 8
    #image = np.array(input[:,:,0],input[:,:,1],input[:,:,2])
    img_bgr =np.array([input[:,:,2],input[:,:,1],input[:,:,0]])
    img_bgr = img_bgr.transpose((1,2,0))
    #img_gray =cv2.cvtColor(np.float32(img_bgr), cv2.COLOR_BGR2GRAY)
    #lbp = local_binary_pattern(img_gray,n_points, radius)
    lbp = np.copy(input)
    for channel in (0,1,2,3):
        lbp[:,:,channel] = local_binary_pattern(input[:,:,channel],n_points,radius)
    #lbp = lbp.transpose((1, 2, 3, 0))
    return lbp


def LEwin3(input):
    LEwin3 = np.copy(input)
    for channel in (0,1,2,3):
        LEwin3[:,:,channel] = entropy(input[:,:,channel],disk(3))
    return LEwin3

def LEwin9(input):
    LEwin9 = np.copy(input)
    for channel in (0,1,2,3):
        LEwin9[:,:,channel] = entropy(input[:,:,channel],disk(9))
    return LEwin9

def LER9_13(input):
    LER9_13 = np.copy(input)
    for channel in (0,1,2,3):
        LER9_13[:,:,channel] = entropy(input[:,:,channel],disk(9))/entropy(input[:,:,channel],disk(13))
    return LER9_13

def LER9_21(input):
    LER9_21 = np.copy(input)
    for channel in (0,1,2,3):
        LER9_21[:,:,channel] = entropy(input[:,:,channel],disk(9))/entropy(input[:,:,channel],disk(21))
    return LER9_21

def getNDVI(data_win):
    R = data_win[:, :, 0]
    NIR = data_win[:, :, 3]
    ndvi = (NIR - R) / (NIR + R)
    np.seterr(divide='ignore', invalid='ignore')
    return ndvi

def getMSAVI(data_win, L=1):  # for sparse vegetation area
    R = data_win[:, :, 0]
    NIR = data_win[:, :, 3]
    # savi = ((NIR - R) / (NIR + R + L)) * (1 + L)
    msavi = (2 * NIR + 1 - np.sqrt((2 * NIR + 1)**2 - 8 * (NIR - R))) / 2
    np.seterr(divide='ignore', invalid='ignore')
    return msavi

def getVARI(data_win):
    R = data_win[:, :, 0]
    G = data_win[:, :, 1]
    B = data_win[:, :, 2]
    vari = (G - R) / (G + R - B)
    np.seterr(divide='ignore', invalid='ignore')
    return vari

def getARVI(data_win):
    R = data_win[:, :, 0]
    B = data_win[:, :, 2]
    NIR = data_win[:, :, 3]
    arvi = (NIR - (2 * R) + B) / (NIR + (2 * R) + B)
    np.seterr(divide='ignore', invalid='ignore')
    return arvi

def getGCI(data_win):
    G = data_win[:, :, 1]
    NIR = data_win[:, :, 3]
    gci = NIR / G - 1
    np.seterr(divide='ignore', invalid='ignore')
    return gci

def getSIPI(data_win):
    R = data_win[:, :, 0]
    B = data_win[:, :, 2]
    NIR = data_win[:, :, 3]
    sipi = (NIR - B) / (NIR - R)
    np.seterr(divide='ignore', invalid='ignore')
    return sipi
<<<<<<< HEAD


=======
>>>>>>> 9ddbe4c2666540c0278ae4209af6620c799a20a3
