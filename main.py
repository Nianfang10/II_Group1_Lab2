import Hand_crafted_features as HCF
from lab2_dataset import SatelliteSet
import torch
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import pickle
from datetime import datetime
import os


def getFeatures(x):
    R = np.expand_dims(x[:, :, 0], axis=-1)
    G = np.expand_dims(x[:, :, 1], axis=-1)
    B = np.expand_dims(x[:, :, 2], axis=-1)
    NIR = np.expand_dims(x[:, :, 3], axis=-1)
    NDVI = np.expand_dims(HCF.getNDVI(x), axis=-1)
    MSAVI = np.expand_dims(HCF.getMSAVI(x), axis=-1)
    VARI = np.expand_dims(HCF.getVARI(x), axis=-1)
    ARVI = np.expand_dims(HCF.getARVI(x), axis=-1)
    GCI = np.expand_dims(HCF.getGCI(x), axis=-1)
    SIPI = np.expand_dims(HCF.getSIPI(x), axis=-1)
    HSV = HCF.HSV(x)
    LAB = HCF.LAB(x)
    SOBEL = HCF.SOBEL(x)
    PREWITT = HCF.PREWITT(x)
    LBP = HCF.LBP(x)

    hcf_list = [R, G, B, NIR, NDVI, MSAVI, VARI, ARVI, GCI, SIPI,
                HSV, LAB, SOBEL, PREWITT, LBP]
    all_features = np.concatenate(hcf_list, axis=2)
    return all_features


# define the dataset and dataloader
dset = SatelliteSet(windowsize=128, split='train')
train_loader = torch.utils.data.DataLoader(dset,
                                           batch_size=8,
                                           num_workers=0,
                                           shuffle=False)

# define the regressors
learning_rate = 'constant'
eta0 = 0.01
reg = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate=learning_rate, eta0=eta0)
scaler_x = StandardScaler()
scaler_y = StandardScaler()

count = 0
for X, Y in tqdm(train_loader):
    # iterate in one small batch
    X = np.transpose(X, (0, 2, 3, 1))

    for i in range(X.shape[0]):
        x = np.asarray(X[i])
        y = Y[i]
        # print(x.shape,y.shape)
        all_features = getFeatures(x)

        # first to standardize features and then start the regression
        all_feat_2D = np.reshape(all_features, (-1, all_features.shape[-1]))  # reshape into a 2D matrix (n*feature numbers) for scaler and regression
        all_feat_2D[np.isinf(all_feat_2D)] = 0
        all_feat_2D[np.isnan(all_feat_2D)] = 0 # replace nan and inf (generated by 0/0) with 0
        y_2D = np.reshape(y, (-1, 1)).numpy().astype(np.float32)
        y_2D[np.isnan(y_2D)] = 0
        y_2D[np.isinf(y_2D)] = 0

        scaler_x.partial_fit(all_feat_2D)
        scaler_y.partial_fit(y_2D)
        std_x = scaler_x.transform(all_feat_2D)
        std_y = scaler_y.transform(y_2D).flatten()
        reg.partial_fit(std_x, std_y)

    count = count + 1
    # save trained model every 50 baches
    if count%50 == 0:
        folder = r'D:\Nianfang\II_Lab\Lab2\data' + os.sep + 'SGD_' + str(learning_rate) + '_' + str(eta0)
        if not os.path.exists(folder):
            os.makedirs(folder)
        model_time = datetime.now().strftime('%m%d-%H-%M-%S')
        pkl_filename = folder + os.sep + 'SGD_' + model_time + '.pkl'
        with open(pkl_filename, 'wb') as file:
            pickle.dump(reg, file)


print("Done fitting")


# load trained model from file
# pkl_filename = ''
# with open(pkl_filename, 'rb') as file:
#     trained_reg = pickle.load(file)


# start predicting
# reg_pre = reg.predict()


