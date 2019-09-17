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
    m = 30
    k = 5
    fs_ratios = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    cv = 5
    metrics = {}
    for fs_ratio in fs_ratios:
        metrics[fs_ratio] = {}
        for clf in ['rf', 'xgb','lr','svm','negboost']:
            metrics[fs_ratio][clf] = {}
            metrics[fs_ratio][clf]['accuracy'] = []
            metrics[fs_ratio][clf]['auc'] = []
            metrics[fs_ratio][clf]['f1'] = []
            metrics[fs_ratio][clf]['recall'] = []

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

        data_train_negative = data_negatives[0][train_negative_index]
        X_db_train = np.concatenate([data_train_positive, data_train_negative])
        Y_db_train = np.concatenate([np.ones((data_train_positive.shape[0])), np.zeros((int(lamda * data_train_positive.shape[0])))])
        for fs_ratio in fs_ratios:
            ns = NegStacking(m=m,fs_ratio=[fs_ratio],lamda=lamda,k=k)
            ns.fit(data_train_positive, data_train_negatives)

            metrics[fs_ratio]['NegStacking']['accuracy'].append(accuracy_score(Y_test, ns.predict(X_test)))
            metrics[fs_ratio]['NegStacking']['auc'].append(roc_auc_score(Y_test, ns.predict_proba(X_test)))
            metrics[fs_ratio]['NegStacking']['f1'].append(f1_score(Y_test, ns.predict(X_test)))
            metrics[fs_ratio]['NegStacking']['recall'].append(recall_score(Y_test, ns.predict(X_test)))
            print('----------------------cv_%d----fs_ratio=%f-----NegStacking---------------------------------' % (cv_index, fs_ratio))
            print('NegStacking accuracy=', metrics[fs_ratio]['NegStacking']['accuracy'][-1])
            print('NegStacking auc=', metrics[fs_ratio]['NegStacking']['auc'][-1])
            print('NegStacking f1=', metrics[fs_ratio]['NegStacking']['f1'][-1])
            print('NegStacking recall=', metrics[fs_ratio]['NegStacking']['recall'][-1])

            select_feature_num = int(1.0 * fs_ratio * X_db_train.shape[1])
            choice_features = np.sort(np.random.choice(X_db_train.shape[1], select_feature_num, replace=False))
            rf = RandomForestClassifier(100).fit(X_db_train[:, choice_features], Y_db_train)
            metrics[fs_ratio]['rf']['accuracy'].append(accuracy_score(Y_test, rf.predict(X_test[:, choice_features])))
            rf_proba = rf.predict_proba(X_test[:, choice_features])[:, 1].flatten()
            metrics[fs_ratio]['rf']['auc'].append(roc_auc_score(Y_test, rf_proba))
            metrics[fs_ratio]['rf']['f1'].append(f1_score(Y_test, rf.predict(X_test[:, choice_features])))
            metrics[fs_ratio]['rf']['recall'].append(recall_score(Y_test, rf.predict(X_test[:, choice_features])))
            print('----------------------cv_%d----fs_ratio=%f-----RF---------------------------------' % (cv_index, fs_ratio))
            print('RF accuracy=', metrics[fs_ratio]['rf']['accuracy'][-1])
            print('RF auc=', metrics[fs_ratio]['rf']['auc'][-1])
            print('RF f1=', metrics[fs_ratio]['rf']['f1'][-1])
            print('RF recall=', metrics[fs_ratio]['rf']['recall'][-1])

            xgb = XGBClassifier(100).fit(X_db_train[:, choice_features], Y_db_train)
            metrics[fs_ratio]['xgb']['accuracy'].append(accuracy_score(Y_test, xgb.predict(X_test[:, choice_features])))
            xgb_proba = xgb.predict_proba(X_test[:, choice_features])[:, 1].flatten()
            metrics[fs_ratio]['xgb']['auc'].append(roc_auc_score(Y_test, xgb_proba))
            metrics[fs_ratio]['xgb']['f1'].append(f1_score(Y_test, xgb.predict(X_test[:, choice_features])))
            metrics[fs_ratio]['xgb']['recall'].append(recall_score(Y_test, xgb.predict(X_test[:, choice_features])))
            print('----------------------cv_%d----fs_ratio=%f-----XGB---------------------------------' % (cv_index, fs_ratio))
            print('XGB accuracy=', metrics[fs_ratio]['xgb']['accuracy'][-1])
            print('XGB auc=', metrics[fs_ratio]['xgb']['auc'][-1])
            print('XGB f1=', metrics[fs_ratio]['xgb']['f1'][-1])
            print('XGB recall=', metrics[fs_ratio]['xgb']['recall'][-1])

            lr = LogisticRegression().fit(X_db_train[:, choice_features], Y_db_train)
            metrics[fs_ratio]['lr']['accuracy'].append(accuracy_score(Y_test, lr.predict(X_test[:, choice_features])))
            lr_proba = lr.predict_proba(X_test[:, choice_features])[:, 1].flatten()
            metrics[fs_ratio]['lr']['auc'].append(roc_auc_score(Y_test, lr_proba))
            metrics[fs_ratio]['lr']['f1'].append(f1_score(Y_test, lr.predict(X_test[:, choice_features])))
            metrics[fs_ratio]['lr']['recall'].append(recall_score(Y_test, lr.predict(X_test[:, choice_features])))
            print('----------------------cv_%d----fs_ratio=%f-----LR---------------------------------' % (cv_index, fs_ratio))
            print('LR accuracy=', metrics[fs_ratio]['lr']['accuracy'][-1])
            print('LR auc=', metrics[fs_ratio]['lr']['auc'][-1])
            print('LR f1=', metrics[fs_ratio]['lr']['f1'][-1])
            print('LR recall=', metrics[fs_ratio]['lr']['recall'][-1])

            svm = SVC(probability=True).fit(X_db_train[:, choice_features], Y_db_train)
            metrics[fs_ratio]['svm']['accuracy'].append(accuracy_score(Y_test, svm.predict(X_test[:, choice_features])))
            svm_proba = svm.predict_proba(X_test[:, choice_features])[:, 1].flatten()
            metrics[fs_ratio]['svm']['auc'].append(roc_auc_score(Y_test, svm_proba))
            metrics[fs_ratio]['svm']['f1'].append(f1_score(Y_test, svm.predict(X_test[:, choice_features])))
            metrics[fs_ratio]['svm']['recall'].append(recall_score(Y_test, svm.predict(X_test[:, choice_features])))
            print('----------------------cv_%d----fs_ratio=%f-----SVM---------------------------------' % (cv_index, fs_ratio))
            print('SVM accuracy=', metrics[fs_ratio]['svm']['accuracy'][-1])
            print('SVM auc=', metrics[fs_ratio]['svm']['auc'][-1])
            print('SVM f1=', metrics[fs_ratio]['svm']['f1'][-1])
            print('SVM recall=', metrics[fs_ratio]['svm']['recall'][-1])

    for fs_ratio in fs_ratios:
        print('---------Summary---NegStacking--fs_ratio=%f----------------------' % fs_ratio)
        print('NegStacking accuracy=', np.mean(metrics[fs_ratio]['NegStacking']['accuracy']))
        print('NegStacking auc=', np.mean(metrics[fs_ratio]['NegStacking']['auc']))
        print('NegStacking f1=', np.mean(metrics[fs_ratio]['NegStacking']['f1']))
        print('NegStacking recall=', np.mean(metrics[fs_ratio]['NegStacking']['recall']))

        print( '----------------------------------Summary---RF--fs_ratio=%f-----------------' % fs_ratio)
        print('RF accuracy=', np.mean(metrics[fs_ratio]['rf']['accuracy']))
        print('RF auc=', np.mean(metrics[fs_ratio]['rf']['auc']))
        print('RF f1=', np.mean(metrics[fs_ratio]['rf']['f1']))
        print('RF recall=', np.mean(metrics[fs_ratio]['rf']['recall']))

        print('----------------------------------Summary---XGB--fs_ratio=%f-----------------' % fs_ratio)
        print('XGB accuracy=', np.mean(metrics[fs_ratio]['xgb']['accuracy']))
        print('XGB auc=', np.mean(metrics[fs_ratio]['xgb']['auc']))
        print('XGB f1=', np.mean(metrics[fs_ratio]['xgb']['f1']))
        print('XGB recall=', np.mean(metrics[fs_ratio]['xgb']['recall']))

        print('----------------------------------Summary---LR--fs_ratio=%f-----------------' % fs_ratio)
        print('LR accuracy=', np.mean(metrics[fs_ratio]['lr']['accuracy']))
        print('LR auc=', np.mean(metrics[fs_ratio]['lr']['auc']))
        print('LR f1=', np.mean(metrics[fs_ratio]['lr']['f1']))
        print('LR recall=', np.mean(metrics[fs_ratio]['lr']['recall']))

        print('----------------------------------Summary---SVM--fs_ratio=%f-----------------' % fs_ratio)
        print('SVM accuracy=', np.mean(metrics[fs_ratio]['svm']['accuracy']))
        print('SVM auc=', np.mean(metrics[fs_ratio]['svm']['auc']))
        print('SVM f1=', np.mean(metrics[fs_ratio]['svm']['f1']))
        print('SVM recall=', np.mean(metrics[fs_ratio]['svm']['recall']))



