from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix, roc_curve, precision_recall_curve, classification_report
import numpy as np

np_data_x, np_data_y = make_classification(
    n_samples=10000, n_features=20, n_informative=4, n_redundant=16, random_state=42
)


rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=42)


lst_rocauc = list()

for idx_train, idx_test in rskf.split(np_data_x, np_data_y):

    np_train_x = np_data_x[idx_train]
    np_train_y = np_data_y[idx_train]
    np_test_x = np_data_x[idx_test]
    np_test_y = np_data_y[idx_test]

    #where to change the model
    lr = LogisticRegression(random_state=42)
    lr.fit(np_train_x, np_train_y)

    #predict probabilities and calculate roc auc
    np_test_y_proba = lr.predict_proba(np_test_x)
    lst_rocauc.append(roc_auc_score(np_test_y, np_test_y_proba[:, 1]))

    #print results
    print('ROC AUC: {:.4f}'.format(lst_rocauc[-1]))
    print(classification_report(np_test_y, lr.predict(np_test_x)))
    print(confusion_matrix(np_test_y, lr.predict(np_test_x)))

print('kfold mean roc auc: {:.4f}'.format(np.mean(lst_rocauc)))
print('kfold mean roc auc: {:.4f}'.format(np.std(lst_rocauc)))
print(lst_rocauc)
