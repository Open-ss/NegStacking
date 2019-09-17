# -*- coding:utf-8 -*-
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score,recall_score
from sklearn.cross_validation import KFold
import utils
from used_models import NegStacking

if __name__ == '__main__':
    run()


def run(path, data_set_name):
    lamda = 1
    ms = [5,10,15,20,25,30,35,40,45,50,55,60,65]
    k = 5
    fs_ratio = [0.6]
    cv = 5
    params = None
    metrics = {}
    combine_methods = ['LR Combine','Tree Combine','NN Combine','XGB Combine','RF Combine','SVM Combine','Avg Combine','Vote Combine']
    
    for m in ms:
        metrics[m] = {}
        for method in combine_methods:
            metrics[m][method] = {}
            metrics[m][method]['accuracy'] = []
            metrics[m][method]['auc'] = []
            metrics[m][method]['f1'] = []
            metrics[m][method]['recall'] = []

    data_positive = np.load(path + data_set_name+ '/data_merge/data_positive.npy')
    np.random.shuffle(data_positive)

    kf = KFold(n=data_positive.shape[0], n_folds=cv, shuffle=True)

    for m in ms:
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

            ns = NegStacking(m=m,fs_ratio=fs_ratio,lamda=lamda,params=params,k=k)
            ns.fit(data_train_positive, data_train_negatives)

            metrics[m]['LR Combine']['accuracy'].append(accuracy_score(Y_test, ns.predict(X_test)))
            metrics[m]['LR Combine']['auc'].append(roc_auc_score(Y_test, ns.predict_proba(X_test)))
            metrics[m]['LR Combine']['f1'].append(f1_score(Y_test, ns.predict(X_test)))
            metrics[m]['LR Combine']['recall'].append(recall_score(Y_test, ns.predict(X_test)))

            metrics[m]['Avg Combine']['accuracy'].append(accuracy_score(Y_test, ns.predict_mean(X_test)))
            metrics[m]['Avg Combine']['auc'].append(roc_auc_score(Y_test, ns.predict_proba_mean(X_test)))
            metrics[m]['Avg Combine']['f1'].append(f1_score(Y_test, ns.predict_mean(X_test)))
            metrics[m]['Avg Combine']['recall'].append(recall_score(Y_test, ns.predict_mean(X_test)))

            metrics[m]['Vote Combine']['accuracy'].append(accuracy_score(Y_test, ns.predict_vote(X_test)))
            metrics[m]['Vote Combine']['auc'].append(roc_auc_score(Y_test, ns.predict_proba_vote(X_test)))
            metrics[m]['Vote Combine']['f1'].append(f1_score(Y_test, ns.predict_vote(X_test)))
            metrics[m]['Vote Combine']['recall'].append(recall_score(Y_test, ns.predict_vote(X_test)))

            metrics[m]['Tree Combine']['accuracy'].append(accuracy_score(Y_test, ns.predict_tree(X_test)))
            metrics[m]['Tree Combine']['auc'].append(roc_auc_score(Y_test, ns.predict_proba_tree(X_test)))
            metrics[m]['Tree Combine']['f1'].append(f1_score(Y_test, ns.predict_tree(X_test)))
            metrics[m]['Tree Combine']['recall'].append(recall_score(Y_test, ns.predict_tree(X_test)))

            metrics[m]['XGB Combine']['accuracy'].append(accuracy_score(Y_test, ns.predict_xgb(X_test)))
            metrics[m]['XGB Combine']['auc'].append(roc_auc_score(Y_test, ns.predict_proba_xgb(X_test)))
            metrics[m]['XGB Combine']['f1'].append(f1_score(Y_test, ns.predict_xgb(X_test)))
            metrics[m]['XGB Combine']['recall'].append(recall_score(Y_test, ns.predict_xgb(X_test)))

            metrics[m]['SVM Combine']['accuracy'].append(accuracy_score(Y_test, ns.predict_svm(X_test)))
            metrics[m]['SVM Combine']['auc'].append(roc_auc_score(Y_test, ns.predict_proba_svm(X_test)))
            metrics[m]['SVM Combine']['f1'].append(f1_score(Y_test, ns.predict_svm(X_test)))
            metrics[m]['SVM Combine']['recall'].append(recall_score(Y_test, ns.predict_svm(X_test)))

            metrics[m]['RF Combine']['accuracy'].append(accuracy_score(Y_test, ns.predict_rf(X_test)))
            metrics[m]['RF Combine']['auc'].append(roc_auc_score(Y_test, ns.predict_proba_rf(X_test)))
            metrics[m]['RF Combine']['f1'].append(f1_score(Y_test, ns.predict_rf(X_test)))
            metrics[m]['RF Combine']['recall'].append(recall_score(Y_test, ns.predict_rf(X_test)))

            metrics[m]['NN Combine']['accuracy'].append(accuracy_score(Y_test, ns.predict_nn(X_test)))
            metrics[m]['NN Combine']['auc'].append(roc_auc_score(Y_test, ns.predict_proba_nn(X_test)))
            metrics[m]['NN Combine']['f1'].append(f1_score(Y_test, ns.predict_nn(X_test)))
            metrics[m]['NN Combine']['recall'].append(recall_score(Y_test, ns.predict_nn(X_test)))

            for method in combine_methods:
                print('-----------------cv_%d----n=%d-----combine method=%s------------------------' % (cv_index, m, method))
                print('accuracy=', metrics[m][method]['accuracy'][-1])
                print('auc=', metrics[m][method]['auc'][-1])
                print('f1=', metrics[m][method]['f1'][-1])
                print('recall=', metrics[m][method]['recall'][-1])

    for m in m:
        for method in combine_methods:
            print('-------------------------Summary-------------combine method=%s-----------n=%d---------------' % (method,m))
            print('accuracy=', np.mean(metrics[m][method]['accuracy']))
            print('auc=', np.mean(metrics[m][method]['auc']))
            print('f1=', np.mean(metrics[m][method]['f1']))
            print('recall=', np.mean(metrics[m][method]['recall']))


