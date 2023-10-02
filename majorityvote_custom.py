from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix, roc_curve, precision_recall_curve, classification_report
import numpy as np
from sklearn.model_selection import train_test_split

import lightgbm as lgb


lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.001,
    'num_leaves': 32,
    'max_depth': 6,
    'min_data_in_leaf': 50,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'feature_fraction': 0.8,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'min_gain_to_split': 0.2,
    'verbose': 0,
    'is_unbalance': True,
    'num_threads': 8,
    'seed': 42,
}


np_data_x, np_data_y = make_classification(
    n_samples=10000, n_features=20, n_informative=4, n_redundant=16, random_state=42
)


rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=42)
rskf2 = RepeatedStratifiedKFold(n_splits=9, n_repeats=1, random_state=42)


lst_rocauc = list()
lst_model = list()
lst_classreport = list()
lst_confusionmatrix = list()

for idx_train, idx_test in rskf.split(np_data_x, np_data_y):

    np_train_x = np_data_x[idx_train]
    np_train_y = np_data_y[idx_train]
    np_test_x = np_data_x[idx_test]
    np_test_y = np_data_y[idx_test]

    # Train a LightGBM model on the inner training set and validate on the inner validation set
    for idx_train2, idx_dev2 in rskf2.split(np_train_x, np_train_y):

        np_train_x2 = np_train_x[idx_train2]
        np_train_y2 = np_train_y[idx_train2]
        np_val_x2 = np_train_x[idx_dev2]
        np_val_y2 = np_train_y[idx_dev2]

        lgb_train = lgb.Dataset(np_train_x2, label=np_train_y2)
        lgb_val = lgb.Dataset(np_val_x2, label=np_val_y2, reference=lgb_train)

        lgbm_0001 = lgb.train(lgbm_params, lgb_train, valid_sets=[lgb_val])
        lst_model.append(lgbm_0001)


    # Loop over the trained LightGBM models and make predictions on the test set based on majority voting
    for c_01 in range(len(lst_model)):

        np_test_x_predproba = lst_model[c_01].predict(np_test_x)
        np_test_x_predclass = np.where(np_test_x_predproba > 0.5, 1, 0)

        if c_01 == 0:
            np_concatclass = np_test_x_predclass.reshape(-1,1)
            np_concatproba = np_test_x_predproba.reshape(-1,1)
        else:
            np_concatclass = np.concatenate((np_concatclass, np_test_x_predclass.reshape(-1,1)), axis=1)
            np_concatproba = np.concatenate((np_concatproba, np_test_x_predproba.reshape(-1,1)), axis=1)

    np_predclass = np.median(np_concatclass, axis=1)
    np_predclass = np_predclass.astype(int)
    np_predproba = np.mean(np_concatproba, axis=1)

    lst_rocauc.append(roc_auc_score(np_test_y, np_predproba))
    lst_classreport.append(classification_report(np_test_y, np_predclass))
    lst_confusionmatrix.append(confusion_matrix(np_test_y, np_predclass))

print('kfold rocauc mean: {:.4f}'.format(np.mean(lst_rocauc)))
print('kfold rocauc std: {:.4f}'.format(np.std(lst_rocauc)))
print(lst_rocauc)
