import Hand_crafted_features as HCF
from lab2_dataset import SatelliteSet
import torch
import h5py
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler


#define the dataset and dataloader
dset = SatelliteSet( windowsize = 128, split='train')
train_loader = torch.utils.data.DataLoader(dset,
                                           batch_size=8,
                                           num_workers=0,
                                           shuffle=False)



# define the regressors
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor 
import xgboost as xgb

SGD_6feat = SGDRegressor()


for X,Y in tqdm(train_loader):
    #iterate in one small batch
    X = np.transpose(X, (0, 2, 3, 1))
    
    for i in range(X.shape[0]):
        
        x = np.asarray(X[i])
        y = Y[i]
        print(x.shape)
        # print(x.shape,y.shape)
        ARVI = np.expand_dims(HCF.getARVI(x), axis=-1)
        NDVI = np.expand_dims(HCF.getNDVI(x), axis=-1)
        
        # x = np.dstack(x,hsv)
        # x = np.dstack(x,lab)
        sobel =  np.expand_dims(HCF.HSV(x), axis=-1)
        # print(sobel.shape)
        # x = np.concatenate([x, ARVI, sobel], axis=-1)
        # print(x.shape)



