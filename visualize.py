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
            pred_label_list.append(predict)
    groud_truth = np.concatenate(true_label_list)
    predict = np.concatenate(pred_label_list)

    rmse = mean_squared_error(groud_truth,predict,squared=False)
    print('RMSE:',rmse)
    return rmse

if __name__ == "__main__":
    model_xgb = xgb.Booster(model_file='../checkpoint/XGBoost/2021-11-16-11-07-32_lr_0.05.json')
    
    #plot the feature importance
    ax = xgb.plot_importance(model_xgb)
    plt.show()
    test_set = SatelliteSet(windowsize=1098, split='test')
    test_loader = torch.utils.data.DataLoader(test_set,
                                            batch_size=8,
                                            num_workers=2,
                                            shuffle=False)
    
    evaluate(test_loader,model_xgb)




    