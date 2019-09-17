# -*- coding:utf-8 -*-
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score,recall_score
from sklearn.cross_validation import KFold
import utils
from sklearn.neural_network import MLPClassifier
from used_models import NegStacking

if __name__ == '__main__':
    run()

def run(path,data_set_name):
    lamda = 1
    m = 30
    ks = [1,2,3,4,5,6,7,8,9,10,12,14]
    fs_ratio = [0.6]
    cv = 5
    params = None
    metrics = {}
    for k in ks:
        metrics[k] = {}
        metrics[k]['accuracy'] = []
        metrics[k]['auc'] = []
        metrics[k]['f1'] = []
        metrics[k]['recall'] = []

    data_positive = np.load(path + data_set_name+ '/data_merge/data_positive.npy')
    np.random.shuffle(data_positive)

    kf = KFold(n=data_positive.shape[0], n_folds=cv, shuffle=True)
    data_negatives = utils.equalization_sampling_data_n_sets(
        path + data_set_name, m, int(lamda * data_positive.shape[0]))

    for cv_index, (train_positive_index, test_positive_index) in enumerate(kf):
        data_train_positive = data_positive[train_positive_index]
        data_test_positive = data_positive[test_positive_index]

        train_negative_index = []
        test_negative_index = []
        for j in range(int(lamda)):
            test_negative_index.append(int(data_positive.shape[0]*j)+test_positive_index)
            train_negative_index.append(int(data_positive.shape[0]*j)+train_positive_index)
        train_negative_index = np.array(train_negative_index).flatten()
        test_negative_index = np.array(test_negative_index).flatten()

        data_train_negatives = []
        for i in range(m):
            data_train_negatives.append(data_negatives[i][train_negative_index])

        data_test_negative = data_negatives[0][test_negative_index]
        X_test = np.concatenate([data_test_positive, data_test_negative])
        Y_test = np.concatenate([np.ones((data_test_positive.shape[0])),np.zeros((int(lamda*data_test_positive.shape[0])))])
        for k in ks:
            ns = NegStacking(m=m,fs_ratio=fs_ratio,lamda=lamda,params=params,k=k)
            ns.fit(data_train_positive, data_train_negatives)

            metrics[k]['accuracy'].append(accuracy_score(Y_test, ns.predict(X_test)))
            metrics[k]['auc'].append(roc_auc_score(Y_test, ns.predict_proba(X_test)))
            metrics[k]['f1'].append(f1_score(Y_test, ns.predict(X_test)))
            metrics[k]['recall'].append(recall_score(Y_test, ns.predict(X_test)))

            print('----------cv_%d----k=%d--------------------------' % (cv_index, k))
            print('NegStacking accuracy=', metrics[k]['accuracy'][-1])
            print('NegStacking auc=', metrics[k]['auc'][-1])
            print('NegStacking f1=', metrics[k]['f1'][-1])
            print('NegStacking recall=', metrics[k]['recall'][-1])

    for k in ks:
        print('---------------Summary-----k=%d-----------------------' % k)
        print('NegStacking accuracy=', np.mean(metrics[k]['accuracy']))
        print('NegStacking auc=', np.mean(metrics[k]['auc']))
        print('NegStacking f1=', np.mean(metrics[k]['f1']))
        print('NegStacking recall=', np.mean(metrics[k]['recall']))



