# -*- coding:utf-8 -*-
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score,recall_score
from sklearn.cross_validation import KFold
import utils
from used_models import NegStacking

if __name__ == '__main__':
    run()

def run(path,data_set_name):
    lamda = 1
    m = 50
    k = 5
    fs_ratio = [0.6]
    cv = 10
    params_list = [
        None,
        {
            'max_depth':[12,13,14,15,16],
        },
        {
            'criterion': ['gini', 'entropy']
        },
        {
            'min_samples_split': [2,3,4],
        },
        {
             'max_depth':[12,13,14,15,16],
             'criterion':['gini','entropy']
        },
        {
            'max_depth':[12,13,14,15,16],
            'min_samples_split': [2,3,4],
        },
        {
            'criterion':['gini','entropy'],
            'min_samples_split': [2,3,4],
        },
        {
            'criterion':['gini','entropy'],
            'max_depth':[12,13,14,15,16],
            'min_samples_split':[2,3,4],
        }
    ]

    metrics = {}
    combine_methods = ['LR','Avg']
    for i in range(len(params_list)):
        metrics[i] = {}
        for combine_method in combine_methods:
            metrics[i][combine_method] = {}
            metrics[i][combine_method]['accuracy'] = []
            metrics[i][combine_method]['auc'] = []
            metrics[i][combine_method]['f1'] = []
            metrics[i][combine_method]['recall'] = []

    data_positive = np.load(path + data_set_name+ '/data_merge/data_positive.npy')
    np.random.shuffle(data_positive)

    kf = KFold(n=data_positive.shape[0], n_folds=cv, shuffle=True)
    data_negatives = utils.equalization_sampling_data_n_sets(path + data_set_name, m, int(lamda * data_positive.shape[0]))

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

        for index,params in enumerate(params_list):
            ns = NegStacking(m,fs_ratio=fs_ratio,lamda=lamda,params=params,k=k)
            ns.fit(data_train_positive, data_train_negatives)

            metrics[index]['LR']['accuracy'].append(accuracy_score(Y_test, ns.predict(X_test)))
            proba = ns.predict_proba(X_test)
            metrics[index]['LR']['auc'].append(roc_auc_score(Y_test, proba))
            metrics[index]['LR']['f1'].append(f1_score(Y_test, ns.predict(X_test)))
            metrics[index]['LR']['recall'].append(recall_score(Y_test, ns.predict(X_test)))

            metrics[index]['Avg']['accuracy'].append(accuracy_score(Y_test, ns.predict_mean(X_test)))
            proba = ns.predict_proba_mean(X_test)
            metrics[index]['Avg']['auc'].append(roc_auc_score(Y_test, proba))
            metrics[index]['Avg']['f1'].append(f1_score(Y_test, ns.predict_mean(X_test)))
            metrics[index]['Avg']['recall'].append(recall_score(Y_test, ns.predict_mean(X_test)))

            for combine_method in combine_methods:
                print('----------------------cv_%d----params=%d----combine_method=%s------------------' % (cv_index, index,combine_method))
                print('accuracy=', metrics[index][combine_method]['accuracy'][-1])
                print('auc=', metrics[index][combine_method]['auc'][-1])
                print('f1=', metrics[index][combine_method]['f1'][-1])
                print('recall=', metrics[index][combine_method]['recall'][-1])

    for index, params in enumerate(params_list):
        for combine_method in combine_methods:
            print('--------------------------------Summary---combine_method=%s--params=%d---------------------------' % (combine_method,index))
            print('accuracy=', np.mean(metrics[index][combine_method]['accuracy']))
            print('auc=', np.mean(metrics[index][combine_method]['auc']))
            print('f1=', np.mean(metrics[index][combine_method]['f1']))
            print('recall=', np.mean(metrics[index][combine_method]['recall']))




