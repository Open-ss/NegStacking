# -*- coding:utf-8 -*-
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score,recall_score
from sklearn.cross_validation import KFold
import utils
from used_models import NegStacking

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

if __name__ == '__main__':
    run()


def run(path, data_set_name):
    m = 30
    k = 5
    fs_ratio = [0.6]
    cv = 5
    lambds = [1,2,3,4,5,6,7]
    params = None

    metrics = {}
    for lambd in lambds:
        metrics[lambd] = {}
        metrics[lambd]['accuracy'] = []
        metrics[lambd]['auc'] = []
        metrics[lambd]['f1'] = []
        metrics[lambd]['recall'] = []

    data_positive = np.load(path + data_set_name+ '/data_merge/data_positive.npy')
    np.random.shuffle(data_positive)

    kf = KFold(n=data_positive.shape[0], n_folds=cv, shuffle=True)

    for lambd in lambds:
        data_negatives = utils.equalization_sampling_data_n_sets(path + data_set_name, m, int(lambd * data_positive.shape[0]))

        for cv_index, (train_positive_index, test_positive_index) in enumerate(kf):
            data_train_positive = data_positive[train_positive_index]
            data_test_positive = data_positive[test_positive_index]

            train_negative_index = []
            test_negative_index = []
            for j in range(int(lambd)):
                test_negative_index.append(int(data_positive.shape[0]*j)+test_positive_index)
                train_negative_index.append(int(data_positive.shape[0]*j)+train_positive_index)
            train_negative_index = np.array(train_negative_index).flatten()
            test_negative_index = np.array(test_negative_index).flatten()

            data_train_negatives = []
            for i in range(m):
                data_train_negatives.append(data_negatives[i][train_negative_index])

            data_test_negative = data_negatives[0][test_negative_index]
            X_test = np.concatenate([data_test_positive, data_test_negative])
            Y_test = np.concatenate([np.ones((data_test_positive.shape[0])),np.zeros((int(lambd*data_test_positive.shape[0])))])

            ns = NegStacking(m=m,fs_ratio=fs_ratio,lamda=lambd,params=params,k=k)
            ns.fit(data_train_positive, data_train_negatives)

            metrics[lambd]['accuracy'].append(accuracy_score(Y_test, ns.predict(X_test)))
            metrics[lambd]['auc'].append(roc_auc_score(Y_test, ns.predict_proba(X_test)))
            metrics[lambd]['f1'].append(f1_score(Y_test, ns.predict(X_test)))
            metrics[lambd]['recall'].append(recall_score(Y_test, ns.predict(X_test)))

            print('----------------------cv_%d----lambd=%f--------------------------------------' % (cv_index, lambd))
            print('NegStacking accuracy=', metrics[lambd]['accuracy'][-1])
            print('NegStacking auc=', metrics[lambd]['auc'][-1])
            print('NegStacking f1=', metrics[lambd]['f1'][-1])
            print('NegStacking recall=', metrics[lambd]['recall'][-1])

    for lambd in lambds:
        print('----------------------------------------------Summary-----lambd=%f----------------------------------------' % lambd)
        print('NegStacking accuracy=', np.mean(metrics[lambd]['accuracy']))
        print('NegStacking auc=', np.mean(metrics[lambd]['auc']))
        print('NegStacking f1=', np.mean(metrics[lambd]['f1']))
        print('NegStacking recall=', np.mean(metrics[lambd]['recall']))



