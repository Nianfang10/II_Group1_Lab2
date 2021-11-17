from lab2_dataset import SatelliteSet
import numpy as np
from tqdm import tqdm
import torch
import pickle
from datetime import datetime
from sklearn.metrics import mean_squared_error

from main import getFeatures
import xgboost as xgb

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
            
            predict[y_2D[:,0]==-1] = -1
            pred_label_list.append(predict)
    groud_truth = np.concatenate(true_label_list)
    predict = np.concatenate(pred_label_list)

    rmse = mean_squared_error(groud_truth,predict,squared=False)
    print('RMSE:',rmse)
    return rmse


def train(dataloader,p_model,params):
   
    for X,Y in tqdm(dataloader):
        x_batch = []
        y_batch = []
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
            y_2D[np.isnan(y_2D)] = -1
            
            y_2D[np.isinf(y_2D)] = -1

            # filter the pixels with no data
            mask = (y_2D!=-1)
            if np.all(mask == False):
                continue
            y_2D = y_2D[mask]
            y_2D = np.reshape(y_2D, (-1, 1))
            all_feat_2D = all_feat_2D[np.repeat(mask,16,axis=1)]
            all_feat_2D = np.reshape(all_feat_2D, (-1, all_features.shape[-1]))
            
            y_batch.append(y_2D)
            x_batch.append(all_feat_2D)
            
        if x_batch == []:
            continue
            
        x_batch = np.concatenate(x_batch)
        y_batch = np.concatenate(y_batch)
        train_data = xgb.DMatrix(x_batch,y_batch,missing=-1)
        model = xgb.train(params,dtrain = train_data,evals=[(train_data,'train')], num_boost_round=50, xgb_model=p_model)
        p_model = model
        
        
        

    return p_model

if __name__ == "__main__":
    dset = SatelliteSet(windowsize=1098, split='train')
    train_loader = torch.utils.data.DataLoader(dset,
                                            batch_size=8,
                                            num_workers=0,
                                            shuffle=False)
    validate_set = SatelliteSet(windowsize=1098, split='validate')
    validate_loader = torch.utils.data.DataLoader(validate_set,
                                            batch_size=8,
                                            num_workers=2,
                                            shuffle=False)

    # xg_reg = xgb.XGBRegressor(max_depth=3, n_estimators=100, n_jobs=2,
    #                        objectvie='reg:squarederror', booster='gbtree',
    #                        random_state=42, learning_rate=0.05)
    # xgb_params = {}

    model = None
    learning_rate = 0.2
    max_depth = 3
    params = {'eta': learning_rate,
            "max_depth": max_depth,
            'refresh_leaf':True,
            'subsample': 0.7}

    
    model = train(train_loader,model,params)

    
    
    
    
    # xgb.plot_importance(model)
    dt=datetime.now()
    date=dt.strftime('%Y-%m-%d-%H-%M-%S')
    filename = '../checkpoint/Xgboost/new_loader' + date +'_lr_'+str(learning_rate)+ 'max_depth'+str(max_depth)+ '.json'
    model.save_model(filename)
    model.load(filename)
    evaluate(validate_loader,model)
    




  
    



    
            

            

