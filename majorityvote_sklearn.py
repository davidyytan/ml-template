from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix, roc_curve, precision_recall_curve, classification_report
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from lightgbm import LGBMClassifier

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


    clf_0001 = LGBMClassifier(**lgbm_params)
    clf_0002 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_0003 = LogisticRegression(random_state=42)

    eclf = VotingClassifier(estimators=[('lgbm', clf_0001), ('rf', clf_0002), ('lr', clf_0003)], voting='soft')

    eclf.fit(np_train_x, np_train_y)

    np_test_y_predproba = eclf.predict_proba(np_test_x)[:, 1]
    np_test_y_preclass = eclf.predict(np_test_x)


    lst_rocauc.append(roc_auc_score(np_test_y, np_test_y_predproba))
    lst_classreport.append(classification_report(np_test_y, np_test_y_preclass))
    lst_confusionmatrix.append(confusion_matrix(np_test_y, np_test_y_preclass))

print('kfold rocauc mean: {:.4f}'.format(np.mean(lst_rocauc)))
print('kfold rocauc std: {:.4f}'.format(np.std(lst_rocauc)))
print(lst_rocauc)
