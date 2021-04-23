import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

X_train = np.load('syllables_wid_wlen_bert_features_train.npy')
y_train = np.load('syllables_wid_wlen_bert_target_train.npy')
X_test = np.load('syllables_wid_wlen_bert_features_test.npy')
y_test = np.load('syllables_wid_wlen_bert_target_test.npy')

X_test.astype('float64')
X_train.astype('float64')

# After trying various runs of grid search the following set of parameter grid values gave good results
lgbm_models = []
# nFix Model
lgbm_models.append(LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
              importance_type='split', lambda_l1=4.6, lambda_l2=8.6,
              learning_rate=0.1, max_depth=-1, min_child_samples=20,
              min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
              n_jobs=-1, num_leaves=75, objective=None, random_state=10,
              reg_alpha=0, reg_lambda=0, silent=True, subsample=1.0,
              subsample_for_bin=200000, subsample_freq=0))
# FFD Model
lgbm_models.append(LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
              importance_type='split', lambda_l1=1.0, lambda_l2=11.6,
              learning_rate=0.1, max_depth=-1, min_child_samples=20,
              min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
              n_jobs=-1, num_leaves=62, objective=None, random_state=10,
              reg_alpha=0.1, reg_lambda=0.0, silent=True, subsample=1.0,
              subsample_for_bin=200000, subsample_freq=0))
# GPT Model
lgbm_models.append(LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
              importance_type='split', lambda_l1=4.6, lambda_l2=9.2,
              learning_rate=0.1, max_depth=-1, min_child_samples=20,
              min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
              n_jobs=-1, num_leaves=62, objective=None, random_state=10,
              reg_alpha=0.1, reg_lambda=0.0, silent=True, subsample=1.0,
              subsample_for_bin=200000, subsample_freq=0))
# TRT Model
lgbm_models.append(LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
              importance_type='split', lambda_l1=9.6, lambda_l2=0.6,
              learning_rate=0.1, max_depth=-1, min_child_samples=20,
              min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
              n_jobs=-1, num_leaves=75, objective=None, random_state=10,
              reg_alpha=0.1, reg_lambda=0.0, silent=True, subsample=1.0,
              subsample_for_bin=200000, subsample_freq=0))
#fixProp
lgbm_models.append(LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
              importance_type='split', lambda_l1=10.6, lambda_l2=18.6,
              learning_rate=0.1, max_depth=-1, min_child_samples=20,
              min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
              n_jobs=-1, num_leaves=31, objective=None, random_state=10,
              reg_alpha=0.1, reg_lambda=0.0, silent=True, subsample=1.0,
              subsample_for_bin=200000, subsample_freq=0))
           
y_pred_lgbm = []
for i in range(5):
  lgbm_models[i].fit(X_train,y_train[:,i])
  y_pred_lgbm.append(lgbm_models[i].predict(X_test))

y_pred_array = np.column_stack(y_pred_lgbm)
mae = []
for i in range(5):
    arr = y_pred_array[:, i]
    mae.append(mean_absolute_error(arr,y_test[:, i]))
