import matplotlib.pyplot as plt
from skimage.feature import greycomatrix
from skimage import data
import cv2
import numpy as np


def getGLCM(data_win):
    imgBGR = np.flip(data_win[:, :, 0:3], -1)  # convert the channels into B, G, R
    imgGray = cv2.cvtColor(np.float32(imgBGR), cv2.COLOR_BGR2GRAY)
    imgGray_int = imgGray.astype('int')  # the greycomatrix function below does not allow "float" type

    # calculate different glcm from different directions
    glcm_h = greycomatrix(imgGray_int, distances=[1], angles=[0], levels=256,
                        symmetric=False)
    glcm_v = greycomatrix(imgGray_int, distances=[1], angles=[np.pi / 2], levels=256,
                        symmetric=False)
    glcm_45 = greycomatrix(imgGray_int, distances=[1], angles=[np.pi / 4], levels=256,
                        symmetric=False)
    h = glcm_h[:, :, 0, 0]
    v = glcm_v[:, :, 0, 0]
    diag = glcm_45[:, :, 0, 0]
    joint = np.array([h, v, diag])  # return a 3*winsize*winsize matrix
    return joint


# data_win = np.random.randint(0, 255, size=(16, 16, 4))
# result = getGLCM(data_win)
