import matplotlib.pyplot as plt


from skimage.feature import greycomatrix, greycoprops
from skimage import data
import cv2
import numpy as np


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


# data_win = data.camera()
# result = getGLCM(data_win)
