import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

X_train = np.load('syllables_bert_features_train.npy')
y_train = np.load('syllables_bert_target_train.npy')
X_test = np.load('syllables_bert_features_test.npy')
y_test = np.load('syllables_bert_target_test.npy')

# After trying various runs of grid search the following set of parameter grid values gave good results
lgbm_models = []
# nFix Model
lgbm_models.append(LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
              importance_type='split', lambda_l1=0.6, lambda_l2=0.6,
              learning_rate=0.1, max_depth=-1, min_child_samples=20,
              min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
              n_jobs=-1, num_leaves=45, objective=None, random_state=10,
              reg_alpha=0, reg_lambda=0, silent=True, subsample=1.0,
              subsample_for_bin=200000, subsample_freq=0))
# FFD Model
lgbm_models.append(LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
              importance_type='split', lambda_l1=0.1, lambda_l2=1.0,
              learning_rate=0.1, max_depth=-1, min_child_samples=20,
              min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
              n_jobs=-1, num_leaves=62, objective=None, random_state=10,
              reg_alpha=0.1, reg_lambda=0.0, silent=True, subsample=1.0,
              subsample_for_bin=200000, subsample_freq=0))
# GPT Model
lgbm_models.append(LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
              importance_type='split', lambda_l1=0.7, lambda_l2=1.4,
              learning_rate=0.1, max_depth=-1, min_child_samples=20,
              min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
              n_jobs=-1, num_leaves=31, objective=None, random_state=10,
              reg_alpha=0.1, reg_lambda=0.0, silent=True, subsample=1.0,
              subsample_for_bin=200000, subsample_freq=0))
# TRT Model
lgbm_models.append(LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
              importance_type='split', lambda_l1=1.6, lambda_l2=0.7,
              learning_rate=0.1, max_depth=-1, min_child_samples=20,
              min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
              n_jobs=-1, num_leaves=62, objective=None, random_state=10,
              reg_alpha=0.1, reg_lambda=0.0, silent=True, subsample=1.0,
              subsample_for_bin=200000, subsample_freq=0))
#fixProp
lgbm_models.append(LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
              importance_type='split', lambda_l1=1.0, lambda_l2=1.0,
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
