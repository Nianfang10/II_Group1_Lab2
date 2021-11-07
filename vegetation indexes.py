import numpy as np

# vegetation indexes
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