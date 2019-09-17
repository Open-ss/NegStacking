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

def run(path,data_set_name):
    lamda = 1
    ms = [50]
    k = 10
    fs_ratio = [0.6]
    cv = 10
    params = {
        'criterion':['gini','entropy'],
        'min_samples_split':[2,3,4],
    }
    metrics = {}
    for m in ms:
        metrics[m] = {}
        for methond in ['NegStacking','RF','AdaBoost','GBDT','XGB','DT','LR','SVM']:
            metrics[m][methond] = {}
            metrics[m][methond]['accuracy'] = []
            metrics[m][methond]['auc'] = []
            metrics[m][methond]['f1'] = []
            metrics[m][methond]['recall'] = []
   
    data_positive = np.load(path + data_set_name+ '/data_merge/data_positive.npy')
    np.random.shuffle(data_positive)

    kf = KFold(n=data_positive.shape[0], n_folds=cv, shuffle=True)

    for m in ms:
        data_negatives = utils.equalization_sampling_data_n_sets(path + data_set_name,m,int(lamda * data_positive.shape[0]))

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

            data_train_negative = data_negatives[0][train_negative_index]

            data_test_negative = data_negatives[0][test_negative_index]
            X_test = np.concatenate([data_test_positive, data_test_negative])
            Y_test = np.concatenate([np.ones((data_test_positive.shape[0])),np.zeros((int(lamda*data_test_positive.shape[0])))])

            ns = NegStacking(m=m,fs_ratio=fs_ratio,lamda=lamda,params=params,k=k)
            ns.fit(data_train_positive, data_train_negatives)

            metrics[m]['NegStacking']['accuracy'].append(accuracy_score(Y_test, ns.predict(X_test)))
            metrics[m]['NegStacking']['auc'].append(roc_auc_score(Y_test, ns.predict_proba(X_test)))
            metrics[m]['NegStacking']['f1'].append(f1_score(Y_test, ns.predict(X_test)))
            metrics[m]['NegStacking']['recall'].append(recall_score(Y_test, ns.predict(X_test)))

            print('----------------------cv_%d----n=%d-----NegStacking---------------------------------' % (cv_index, m))
            print('NegStacking accuracy=', metrics[m]['NegStacking']['accuracy'][-1])
            print('NegStacking auc=', metrics[m]['NegStacking']['auc'][-1])
            print('NegStacking f1=', metrics[m]['NegStacking']['f1'][-1])
            print('NegStacking recall=', metrics[m]['NegStacking']['recall'][-1])

            print('----------------------cv_%d----n=%d-----RF---------------------------------' % (cv_index, m))
            X_db_train = np.concatenate([data_train_positive, data_train_negative])
            Y_db_train = np.concatenate([np.ones((data_train_positive.shape[0])),np.zeros((int(lamda*data_train_positive.shape[0])))])
            rf = RandomForestClassifier(100).fit(X_db_train,Y_db_train)
            metrics[m]['RF']['accuracy'].append(accuracy_score(Y_test, rf.predict(X_test)))
            metrics[m]['RF']['auc'].append(roc_auc_score(Y_test, rf.predict_proba(X_test)[:,1].flatten()))
            metrics[m]['RF']['f1'].append(f1_score(Y_test, rf.predict(X_test)))
            metrics[m]['RF']['recall'].append(recall_score(Y_test, rf.predict(X_test)))
            print('RF accuracy=', metrics[m]['RF']['accuracy'][-1])
            print('RF auc=', metrics[m]['RF']['auc'][-1])
            print('RF f1=', metrics[m]['RF']['f1'][-1])
            print('RF recall=', metrics[m]['RF']['recall'][-1])

            print( '----------------------cv_%d----n=%d-----GBDT---------------------------------' % (cv_index, m))
            gbdt = GradientBoostingClassifier().fit(X_db_train,Y_db_train)
            metrics[m]['GBDT']['accuracy'].append(accuracy_score(Y_test, gbdt.predict(X_test)))
            metrics[m]['GBDT']['auc'].append(roc_auc_score(Y_test, gbdt.predict_proba(X_test)[:, 1].flatten()))
            metrics[m]['GBDT']['f1'].append(f1_score(Y_test, gbdt.predict(X_test)))
            metrics[m]['GBDT']['recall'].append(recall_score(Y_test, gbdt.predict(X_test)))
            print('GBDT accuracy=', metrics[m]['GBDT']['accuracy'][-1])
            print('GBDT auc=', metrics[m]['GBDT']['auc'][-1])
            print('GBDT f1=', metrics[m]['GBDT']['f1'][-1])
            print('GBDT recall=', metrics[m]['GBDT']['recall'][-1])

            print('----------------------cv_%d----n=%d-----XGB---------------------------------' % (cv_index, m))
            xgb = XGBClassifier().fit(X_db_train,Y_db_train)
            metrics[m]['XGB']['accuracy'].append(accuracy_score(Y_test, xgb.predict(X_test)))
            metrics[m]['XGB']['auc'].append(roc_auc_score(Y_test, xgb.predict_proba(X_test)[:, 1].flatten()))
            metrics[m]['XGB']['f1'].append(f1_score(Y_test, xgb.predict(X_test)))
            metrics[m]['XGB']['recall'].append(recall_score(Y_test, xgb.predict(X_test)))
            print('XGB accuracy=', metrics[m]['XGB']['accuracy'][-1])
            print('XGB auc=', metrics[m]['XGB']['auc'][-1])
            print('XGB f1=', metrics[m]['XGB']['f1'][-1])
            print('XGB recall=', metrics[m]['XGB']['recall'][-1])

            print('----------------------cv_%d----n=%d-----DT---------------------------------' % (cv_index, m))
            dt = DecisionTreeClassifier().fit(X_db_train,Y_db_train)
            metrics[m]['DT']['accuracy'].append(accuracy_score(Y_test, dt.predict(X_test)))
            metrics[m]['DT']['auc'].append(roc_auc_score(Y_test, dt.predict_proba(X_test)[:, 1].flatten()))
            metrics[m]['DT']['f1'].append(f1_score(Y_test, dt.predict(X_test)))
            metrics[m]['DT']['recall'].append(recall_score(Y_test, dt.predict(X_test)))
            print('DT accuracy=', metrics[m]['DT']['accuracy'][-1])
            print('DT auc=', metrics[m]['DT']['auc'][-1])
            print('DT f1=', metrics[m]['DT']['f1'][-1])
            print('DT recall=', metrics[m]['DT']['recall'][-1])

            print('----------------------cv_%d----n=%d-----LR---------------------------------' % (cv_index, m))
            lr = LogisticRegression().fit(X_db_train,Y_db_train)
            metrics[m]['LR']['accuracy'].append(accuracy_score(Y_test, lr.predict(X_test)))
            metrics[m]['LR']['auc'].append(roc_auc_score(Y_test, lr.predict_proba(X_test)[:, 1].flatten()))
            metrics[m]['LR']['f1'].append(f1_score(Y_test, lr.predict(X_test)))
            metrics[m]['LR']['recall'].append(recall_score(Y_test, lr.predict(X_test)))
            print('LR accuracy=', metrics[m]['LR']['accuracy'][-1])
            print('LR auc=', metrics[m]['LR']['auc'][-1])
            print('LR f1=', metrics[m]['LR']['f1'][-1])
            print('LR recall=', metrics[m]['LR']['recall'][-1])

            print('----------------------cv_%d----n=%d-----SVM---------------------------------' % (cv_index, m))
            svm = SVC(probability=True).fit(X_db_train,Y_db_train)
            metrics[m]['SVM']['accuracy'].append(accuracy_score(Y_test, svm.predict(X_test)))
            metrics[m]['SVM']['auc'].append(roc_auc_score(Y_test, svm.predict_proba(X_test)[:, 1].flatten()))
            metrics[m]['SVM']['f1'].append(f1_score(Y_test, svm.predict(X_test)))
            metrics[m]['SVM']['recall'].append(recall_score(Y_test, svm.predict(X_test)))
            print('SVM accuracy=', metrics[m]['SVM']['accuracy'][-1])
            print('SVM auc=', metrics[m]['SVM']['auc'][-1])
            print('SVM f1=', metrics[m]['SVM']['f1'][-1])
            print('SVM recall=', metrics[m]['SVM']['recall'][-1])

            print('----------------------cv_%d----n=%d-----AdaBoost---------------------------------' % (cv_index, m))
            adb = AdaBoostClassifier(n_estimators=100,base_estimator=DecisionTreeClassifier()).fit(X_db_train, Y_db_train)
            metrics[m]['AdaBoost']['accuracy'].append(accuracy_score(Y_test, adb.predict(X_test)))
            metrics[m]['AdaBoost']['auc'].append(
                roc_auc_score(Y_test, adb.predict_proba(X_test)[:, 1].flatten()))
            metrics[m]['AdaBoost']['f1'].append(f1_score(Y_test, adb.predict(X_test)))
            metrics[m]['AdaBoost']['recall'].append(recall_score(Y_test, adb.predict(X_test)))
            print('AdaBoost accuracy=', metrics[m]['AdaBoost']['accuracy'][-1])
            print('AdaBoost auc=', metrics[m]['AdaBoost']['auc'][-1])
            print('AdaBoost f1=', metrics[m]['AdaBoost']['f1'][-1])
            print('AdaBoost recall=', metrics[m]['AdaBoost']['recall'][-1])

    for m in ms:
        print('-----------------------------------Summary---NegStacking--n=%d-------------------------------' % m)
        print('NegStacking accuracy=', np.mean(metrics[m]['NegStacking+LR']['accuracy']))
        print('NegStacking auc=', np.mean(metrics[m]['NegStacking+LR']['auc']))
        print('NegStacking f1=', np.mean(metrics[m]['NegStacking+LR']['f1']))
        print('NegStacking recall=', np.mean(metrics[m]['NegStacking+LR']['recall']))

        print('------------------------------------Summary---RF--n=%d----------------------------------------' % m)
        print('RF accuracy=', np.mean(metrics[m]['RF']['accuracy']))
        print('RF auc=', np.mean(metrics[m]['RF']['auc']))
        print('RF f1=', np.mean(metrics[m]['RF']['f1']))
        print('RF recall=', np.mean(metrics[m]['RF']['recall']))

        print('--------------------------------Summary---AdaBoost--n=%d----------------------------------------' % m)
        print('AdaBoost accuracy=', np.mean(metrics[m]['AdaBoost']['accuracy']))
        print('AdaBoost auc=', np.mean(metrics[m]['AdaBoost']['auc']))
        print('AdaBoost f1=', np.mean(metrics[m]['AdaBoost']['f1']))
        print('AdaBoost recall=', np.mean(metrics[m]['AdaBoost']['recall']))

        print('------------------------------------Summary---GBDT--n=%d----------------------------------------' % m)
        print('GBDT accuracy=', np.mean(metrics[m]['GBDT']['accuracy']))
        print('GBDT auc=', np.mean(metrics[m]['GBDT']['auc']))
        print('GBDT f1=', np.mean(metrics[m]['GBDT']['f1']))
        print('GBDT recall=', np.mean(metrics[m]['GBDT']['recall']))

        print('-------------------------------------Summary---XGB--n=%d----------------------------------------' % m)
        print('XGB accuracy=', np.mean(metrics[m]['XGB']['accuracy']))
        print('XGB auc=', np.mean(metrics[m]['XGB']['auc']))
        print('XGB f1=', np.mean(metrics[m]['XGB']['f1']))
        print('XGB recall=', np.mean(metrics[m]['XGB']['recall']))

        print('-----------------------------------Summary---DT--n=%d----------------------------------------' % m)
        print('DT accuracy=', np.mean(metrics[m]['DT']['accuracy']))
        print('DT auc=', np.mean(metrics[m]['DT']['auc']))
        print('DT f1=', np.mean(metrics[m]['DT']['f1']))
        print('DT recall=', np.mean(metrics[m]['DT']['recall']))

        print('------------------------------------Summary---LR--n=%d----------------------------------------' % m)
        print('LR accuracy=', np.mean(metrics[m]['LR']['accuracy']))
        print('LR auc=', np.mean(metrics[m]['LR']['auc']))
        print('LR f1=', np.mean(metrics[m]['LR']['f1']))
        print('LR recall=', np.mean(metrics[m]['LR']['recall']))

        print('------------------------------------Summary---SVM--n=%d----------------------------------------' % m)
        print('SVM accuracy=', np.mean(metrics[m]['SVM']['accuracy']))
        print('SVM auc=', np.mean(metrics[m]['SVM']['auc']))
        print('SVM f1=', np.mean(metrics[m]['SVM']['f1']))
        print('SVM recall=', np.mean(metrics[m]['SVM']['recall']))
