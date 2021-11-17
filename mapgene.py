from lab2_dataset import SatelliteSet
import numpy as np
from tqdm import tqdm
import torch

from datetime import datetime
import os
from sklearn.metrics import mean_squared_error

from main import getFeatures
import xgboost as xgb
import matplotlib.pyplot as plt
import h5py
import skimage.transform as transform


def evaluate(dataloader,model):
    print('\nstart evaluate\n')
    
    true_label_list = []
    pred_label_list = []
    for X,Y in tqdm(dataloader):
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
            
            true_label_list.append(y_2D)
            predict = model.predict(xgb.DMatrix(all_feat_2D))
            predict = np.reshape(predict,(-1,1))
            predict[y_2D==-1]=-1
            pred_label_list.append(predict)
            
            # plt.rcParams["figure.figsize"] = (10,6)
            # f,axarr = plt.subplots(ncols=2, nrows=1)
            # axarr[0].set_title('Prediction')
            # axarr[0].imshow(np.reshape(predict,(X.shape[1],X.shape[2])))
            # axarr[1].set_title('Ground Truth')
            # axarr[1].imshow(y)
            # plt.show()


    groud_truth = np.concatenate(true_label_list)
    predict = np.concatenate(pred_label_list)

    rmse = mean_squared_error(groud_truth,predict,squared=False)
    print('RMSE:',rmse)
    return rmse

if __name__ == "__main__":
    model = xgb.Booster(model_file='../checkpoint/XGBoost/new_loader2021-11-17-08-56-09_lr_0.2_max_depth3.json')
    
    #plot the feature importance
    # ax = xgb.plot_importance(model_xgb)
    # plt.show()
    test_set = SatelliteSet(windowsize=1098, split='test')
    test_loader = torch.utils.data.DataLoader(test_set,
                                            batch_size=8,
                                            num_workers=2,
                                            shuffle=False)
    
    # evaluate(test_loader,model_xgb)
    dset = h5py.File("../data\dataset_rgb_nir_test.hdf5", "r")
    CLD = dset['CLD_0']
    GT = dset['GT'][0]
    RGB = dset['INPT_0']
    NIR = dset['NIR_0']


    # CLD_mask = CLD==0
    # NIR = NIR * (CLD_mask)
    # RGB = RGB * np.stack((CLD_mask,CLD_mask,CLD_mask),axis = 3)

    # loop over the images through small windows
    window_size = 1098
    image_size = 10980
    num = int(image_size/window_size)
    count = 0
    final_pred = np.zeros((image_size,image_size))
    for i in tqdm(range(num)):
        for j in range(num):

            rgb = np.asarray(RGB[:, i*1098:i*1098+1098, j*1098:j*1098+1098])
            nir = np.asarray(NIR[:, i*1098:i*1098+1098, j*1098:j*1098+1098])
            cld = np.asarray(CLD[:, i*1098:i*1098+1098, j*1098:j*1098+1098])
            gt = np.asarray(GT[ i*1098:i*1098+1098, j*1098:j*1098+1098])
            cld_mask = cld == 0
            nir = nir * cld_mask
            rgb = rgb * np.stack((cld_mask, cld_mask, cld_mask), axis=3)
            rgb_tsum = np.sum((rgb != 0).astype(float), 0)
            rgb_tsum[rgb_tsum == 0] = 1
            rgb = np.sum(rgb, 0)/rgb_tsum

            nir_tsum = np.sum((nir != 0).astype(float), 0)
            nir_tsum[nir_tsum == 0] = 1
            nir = np.sum(nir, 0)/nir_tsum

            # print(rgb.shape, nir.shape)
            x = np.concatenate([rgb, np.expand_dims(nir, axis=-1)], axis=-1)

            all_features = getFeatures(x)


            all_feat_2D = np.reshape(all_features, (
            -1, all_features.shape[-1]))  # reshape into a 2D matrix (n*feature numbers) for scaler and regression
            all_feat_2D[np.isinf(all_feat_2D)] = 0
            all_feat_2D[np.isnan(all_feat_2D)] = 0  # replace nan and inf (generated by 0/0) with 0
            pred = model.predict(xgb.DMatrix(all_feat_2D))
            pred = np.reshape(pred,(window_size,window_size))
            pred[gt==-1]=-1
            final_pred[i*1098:i*1098+1098, j*1098:j*1098+1098] = pred

    plt.imshow(final_pred)
    plt.savefig('../checkpoint/'+'map2.JPG')













    
    








    