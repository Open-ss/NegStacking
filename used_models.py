# -*- coding:utf-8 -*-
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

class NegStacking(object):
    def __init__(self,m,fs_ratio=[1],lamda=1,params=None,k=10,random_state=None):
        self.m = m
        self.lamda = lamda
        self.fs_ratio = fs_ratio
        self.params = params
        self.clfs = []
        self.clfs2 = []
        self.k = k
        self.random_state = random_state
    def fit(self,X_positive,X_negatives):
        # if len(X_negatives)!=self.m:
        #     raise Exception('len of X_negatives is not correct')
        if int(self.lamda * X_positive.shape[0])!= X_negatives[0].shape[0]:
            raise Exception('shape of X_negative is not correct')

        kf = None
        np.random.seed(self.random_state)
        if self.k>1:
            kf = KFold(n=X_positive.shape[0], n_folds=self.k,shuffle=True)

        for i in range(self.m):
            if self.random_state != None:
                self.random_state = self.random_state + 10
            clf = DT(self.k,self.fs_ratio,self.lamda,self.params,self.random_state)
            clf.fit(X_positive,X_negatives[i],kf)
            self.clfs.append(clf)

        X_second_learn = np.zeros((int((1+self.lamda)*X_positive.shape[0]),self.m))
        Y_second_learn = np.zeros(int((1+self.lamda)*X_positive.shape[0]))
        second_start_index = 0
        if self.random_state != None:
            np.random.seed(self.random_state)
            self.random_state = self.random_state + 10
        select_test_index = np.random.randint(0, self.m)

        if self.k==1:
            X_test = np.concatenate([X_positive,X_negatives[select_test_index]])
            Y_test = np.concatenate([np.ones(X_positive.shape[0]),np.zeros(self.lamda*X_positive.shape[0])])
            for i in range(self.m):
                X_second_learn[second_start_index:second_start_index + X_test.shape[0], i] =  self.clfs[i].clfs[0].predict_proba(X_test[:,self.clfs[i].choice_features])[:, 1]
            Y_second_learn[second_start_index:second_start_index + X_test.shape[0]] = Y_test
        else:
            for cv_index, (train_positive_index, test_positive_index) in enumerate(kf):
                data_test_positive = X_positive[test_positive_index]
                test_negative_index = []
                for j in range(int(self.lamda)):
                    test_negative_index.append(int(X_positive.shape[0] * j) + test_positive_index)
                test_negative_index = np.array(test_negative_index).flatten()
                data_test_negative = X_negatives[select_test_index][test_negative_index]
                X_test = np.concatenate([data_test_positive, data_test_negative])
                Y_test = np.concatenate([np.ones(data_test_positive.shape[0]),np.zeros(int(self.lamda * data_test_positive.shape[0]))])

                for i in range(self.m):
                    X_second_learn[second_start_index:second_start_index + X_test.shape[0], i] =  self.clfs[i].clfs[cv_index].predict_proba(X_test[:,self.clfs[i].choice_features])[:, 1]
                    # X_second_learn[second_start_index:second_start_index + X_test.shape[0], i] = self.clfs[i].clfs[cv_index].predict(X_test[:, self.clfs[i].choice_features])
                Y_second_learn[second_start_index:second_start_index + X_test.shape[0]] = Y_test
                second_start_index = second_start_index + X_test.shape[0]

        if self.random_state != None:
            self.random_state = self.random_state + 10
        # self.second_clf = LogisticRegression(max_iter=150,C=0.8,random_state=self.random_state).fit(X_second_learn,Y_second_learn)
        self.second_clf = LogisticRegression(max_iter=150, C=0.8,multi_class='multinomial',solver='newton-cg',random_state=self.random_state).fit(X_second_learn,Y_second_learn)

        self.second_clf_tree = DecisionTreeClassifier(random_state=self.random_state).fit(X_second_learn,Y_second_learn)
        self.second_clf_nn = MLPClassifier(random_state=self.random_state,hidden_layer_sizes=(1+self.m//2,)).fit(X_second_learn,Y_second_learn)
        self.second_clf_xgb = XGBClassifier(seed=self.random_state).fit(X_second_learn,Y_second_learn)
        self.second_clf_rf = RandomForestClassifier(random_state=self.random_state).fit(X_second_learn, Y_second_learn)
        self.second_clf_svm = SVC(random_state=self.random_state,probability=True).fit(X_second_learn, Y_second_learn)
        for i in range(self.m):
            if self.random_state != None:
                self.random_state = self.random_state + 10
            if self.params == None:
                clf = DecisionTreeClassifier(random_state=self.random_state)
            else:
                clf = DecisionTreeClassifier(random_state=self.random_state,**self.clfs[i].parm_dic)
            #clf = DecisionTreeClassifier(random_state=self.random_state)
            X = np.concatenate([X_positive, X_negatives[i]])
            Y = np.concatenate([np.ones(X_positive.shape[0]), np.zeros(int(self.lamda * X_positive.shape[0]))])
            clf.fit(X[:, self.clfs[i].choice_features], Y)
            self.clfs2.append(clf)
    def predict_proba(self,X_test):
        preds = np.zeros((X_test.shape[0],self.m))
        for i in range(self.m):
            preds[:,i] = (self.clfs[i].predict_proba(X_test))
        return self.second_clf.predict_proba(preds)[:,1].flatten()
    def predict(self,X_test):
        preds = self.predict_proba(X_test)
        pred_label = np.zeros(preds.shape[0])
        pred_label[preds >= 0.5] = 1
        return pred_label
    def predict_proba_mean(self,X_test):
        preds = []
        for i in range(self.m):
            pred = self.clfs2[i].predict_proba(X_test[:, self.clfs[i].choice_features])[:,1].flatten()
            #pred = self.clfs[i].predict_proba(X_test)
            preds.append(list(pred))
        preds = np.array(preds)
        preds = np.mean(preds, axis=0)
        return preds
    def predict_mean(self,X_test):
        preds = self.predict_proba_mean(X_test)
        pred_label = np.zeros(preds.shape[0])
        pred_label[preds >= 0.5] = 1
        return pred_label
    def predict_proba_vote(self,X_test):
        preds = []
        for i in range(self.m):
            pred = self.clfs2[i].predict(X_test[:, self.clfs[i].choice_features])
            preds.append(list(pred))
        preds = np.array(preds)
        preds = np.mean(preds, axis=0)
        return preds
    def predict_vote(self,X_test):
        preds = self.predict_proba_vote(X_test)
        pred_label = np.zeros(preds.shape[0])
        pred_label[preds >= 0.5] = 1
        return pred_label

    def predict_proba_tree(self, X_test):
        preds = np.zeros((X_test.shape[0], self.m))
        for i in range(self.m):
            preds[:, i] = (self.clfs[i].predict_proba(X_test))
        return self.second_clf_tree.predict_proba(preds)[:, 1].flatten()
    def predict_tree(self, X_test):
        preds = self.predict_proba_tree(X_test)
        pred_label = np.zeros(preds.shape[0])
        pred_label[preds >= 0.5] = 1
        return pred_label

    def predict_proba_nn(self, X_test):
        preds = np.zeros((X_test.shape[0], self.m))
        for i in range(self.m):
            preds[:, i] = (self.clfs[i].predict_proba(X_test))
        return self.second_clf_nn.predict_proba(preds)[:, 1].flatten()
    def predict_nn(self, X_test):
        preds = self.predict_proba_nn(X_test)
        pred_label = np.zeros(preds.shape[0])
        pred_label[preds >= 0.5] = 1
        return pred_label

    def predict_proba_xgb(self, X_test):
        preds = np.zeros((X_test.shape[0], self.m))
        for i in range(self.m):
            preds[:, i] = (self.clfs[i].predict_proba(X_test))
        return self.second_clf_xgb.predict_proba(preds)[:, 1].flatten()
    def predict_xgb(self, X_test):
        preds = self.predict_proba_xgb(X_test)
        pred_label = np.zeros(preds.shape[0])
        pred_label[preds >= 0.5] = 1
        return pred_label

    def predict_proba_rf(self, X_test):
        preds = np.zeros((X_test.shape[0], self.m))
        for i in range(self.m):
            preds[:, i] = (self.clfs[i].predict_proba(X_test))
        return self.second_clf_rf.predict_proba(preds)[:, 1].flatten()
    def predict_rf(self, X_test):
        preds = self.predict_proba_rf(X_test)
        pred_label = np.zeros(preds.shape[0])
        pred_label[preds >= 0.5] = 1
        return pred_label

    def predict_proba_svm(self, X_test):
        preds = np.zeros((X_test.shape[0], self.m))
        for i in range(self.m):
            preds[:, i] = (self.clfs[i].predict_proba(X_test))
        return self.second_clf_svm.predict_proba(preds)[:, 1].flatten()
    def predict_svm(self, X_test):
        preds = self.predict_proba_svm(X_test)
        pred_label = np.zeros(preds.shape[0])
        pred_label[preds >= 0.5] = 1
        return pred_label
    # pass
class DT(object):
    def __init__(self,k,fs_ratio=[1],lamda=1,params=None,random_state=None):
        self.lamda = lamda
        self.fs_ratio = fs_ratio
        self.params = params
        self.clfs = []
        self.k = k
        self.random_state = random_state
    def fit(self,X_positive,X_negative,kf):
        if int(self.lamda * X_positive.shape[0])!= X_negative.shape[0]:
            raise Exception('shape of X_negative is not correct')
        if kf!=None:
            if kf.n!=X_positive.shape[0]:
                raise Exception('kf.n is not match shape of X_positive')
            if len(kf)!=self.k:
                raise Exception('len of kf is not match k')
        if self.params != None:
            parm_dic = {}
            for name, parm in self.params.items():
                if self.random_state != None:
                    np.random.seed(self.random_state)
                    self.random_state = self.random_state + 10
                parm_dic[name] = parm[np.random.randint(len(parm))]
            self.parm_dic = parm_dic
        if self.random_state != None:
            np.random.seed(self.random_state)
            self.random_state = self.random_state + 10
        select_feature_num = int(1.0*self.fs_ratio[np.random.randint(len(self.fs_ratio))] * X_positive.shape[1])
        if self.random_state != None:
            np.random.seed(self.random_state)
            self.random_state = self.random_state + 10
        self.choice_features = np.sort(np.random.choice(X_positive.shape[1], select_feature_num, replace=False))
        if self.k==1:
            if self.random_state != None:
                self.random_state = self.random_state + 10
            if self.params == None:
                clf = DecisionTreeClassifier(random_state=self.random_state)
            else:
                clf = DecisionTreeClassifier(random_state=self.random_state, **self.parm_dic)
            X = np.concatenate([X_positive, X_negative])
            Y = np.zeros(X.shape[0])
            Y[:X_positive.shape[0]] = 1
            clf.fit(X[:, self.choice_features], Y)
            self.clfs.append(clf)
        else:
            for cv_index, (train_positive_index, test_positive_index) in enumerate(kf):
                data_train_positive = X_positive[train_positive_index]

                train_negative_index = []
                for j in range(int(self.lamda)):
                    train_negative_index.append(int(X_positive.shape[0] * j) + train_positive_index)
                train_negative_index = np.array(train_negative_index).flatten()

                data_train_negative = X_negative[[train_negative_index]]

                if self.random_state != None:
                    self.random_state = self.random_state + 10
                if self.params == None:
                    clf = DecisionTreeClassifier(random_state=self.random_state)
                else:
                    clf = DecisionTreeClassifier(random_state=self.random_state, **self.parm_dic)

                X = np.concatenate([data_train_positive, data_train_negative])
                Y = np.zeros(X.shape[0])
                Y[:data_train_positive.shape[0]] = 1

                clf.fit(X[:, self.choice_features], Y)
                self.clfs.append(clf)
    def predict_proba(self,X_test):
        preds = []
        for i in range(self.k):
            pred = (self.clfs[i].predict_proba(X_test[:,self.choice_features]))[:, 1].flatten()
            # pred = (self.clfs[i].predict(X_test[:, self.choice_features]))
            preds.append(list(pred))
        preds = np.array(preds)
        preds = np.mean(preds, axis=0)
        return preds
    def predict(self,X_test):
        preds = self.predict_proba(X_test)
        pred_label = np.zeros(preds.shape[0])
        pred_label[preds >= 0.5] = 1
        return pred_label
    pass

class EnsemRF(object):
    def __init__(self,m,fs_ratio=1,lamda=1,params=None,k=10,random_state=None):
        self.m = m
        self.lamda = lamda
        self.fs_ratio = fs_ratio
        self.params = params
        self.clfs = []
        self.k = k
        self.n_choice_features = []
        self.random_state = random_state
    def fit(self,X_positive,X_negatives):
        if len(X_negatives)!=self.m:
            raise Exception('len of X_negatives is not correct')
        if int(self.lamda * X_positive.shape[0])!= X_negatives[0].shape[0]:
            raise Exception('shape of X_negative is not correct')
        select_feature_num = int(1.0 * X_positive.shape[1] * self.fs_ratio)
        for i in range(self.m):
            if self.random_state != None:
                self.random_state = self.random_state + 10
            clf = RandomForestClassifier(n_estimators=self.k,max_depth=10,random_state=self.random_state)
            X = np.concatenate([X_positive,X_negatives[i]])
            Y = np.zeros(X.shape[0])
            Y[:X_positive.shape[0]] = 1
            if self.random_state != None:
                self.random_state = self.random_state + 10
            np.random.seed(self.random_state)
            choice_features = np.sort(np.random.choice(X_positive.shape[1], select_feature_num, replace=False))
            self.n_choice_features.append(choice_features)
            clf.fit(X[:,self.n_choice_features[i]],Y)
            self.clfs.append(clf)
    def predict_proba(self,X_test):
        preds = []
        for i in range(self.m):
            pred = (self.clfs[i].predict_proba(X_test[:,self.n_choice_features[i]]))[:, 1].flatten()
            preds.append(list(pred))
        preds = np.array(preds)
        preds = np.mean(preds, axis=0)
        return preds
    def predict(self,X_test):
        preds = self.predict_proba(X_test)
        pred_label = np.zeros(preds.shape[0])
        pred_label[preds >= 0.5] = 1
        return pred_label
    pass

class DT_V2(object):
    def __init__(self,k,fs_ratio=1,params=None,random_state=None,reduce_dim_ratio=0.7):
        self.fs_ratio = fs_ratio
        self.params = params
        self.clfs = []
        self.k = k
        self.random_state = random_state
        self.reduce_dim_ratio = reduce_dim_ratio
    def fit(self,X,Y,kf,drug_data,tp_data):
        if kf.n!=X.shape[0]:
            raise Exception('kf.n is not match shape of X')
        if len(kf)!=self.k:
            raise Exception('len of kf is not match k')
        if self.params != None:
            parm_dic = {}
            for name, parm in self.params.items():
                if self.random_state != None:
                    np.random.seed(self.random_state)
                    self.random_state = self.random_state + 10
                parm_dic[name] = parm[np.random.randint(len(parm))]
            self.parm_dic = parm_dic

        if self.random_state != None:
            self.random_state = self.random_state + 10
            np.random.seed(self.random_state)
        self.choice_feature_drug = np.sort(np.random.choice(X.shape[1] / 2, int(0.5 * self.fs_ratio * X.shape[1]),replace=False))
        if self.random_state != None:
            self.random_state = self.random_state + 10
            np.random.seed(self.random_state)
        self.choice_feature_tp = np.sort(np.random.choice(range(X.shape[1] / 2, X.shape[1]),int(0.5 * self.fs_ratio * X.shape[1]),replace=False))

        X_drug = X[:, self.choice_feature_drug]
        X_tp = X[:, self.choice_feature_tp]

        if self.reduce_dim_ratio<1:
            self.drug_svd = TruncatedSVD(n_components=int(X_drug.shape[1] * self.reduce_dim_ratio)).fit(drug_data[:, self.choice_feature_drug])
            self.tp_svd = TruncatedSVD(n_components=int(X_tp.shape[1] * self.reduce_dim_ratio)).fit(tp_data[:, self.choice_feature_tp - (X.shape[1] / 2)])

            X_drug = self.drug_svd.transform(X_drug)
            X_tp = self.tp_svd.transform(X_tp)
        X = np.concatenate([X_drug, X_tp], axis=1)
        for cv_index, (train_index, test_index) in enumerate(kf):
            X_train = X[train_index]
            Y_train = Y[train_index]

            if self.random_state != None:
                self.random_state = self.random_state + 10
            if self.params == None:
                clf = DecisionTreeClassifier(random_state=self.random_state)
            else:
                clf = DecisionTreeClassifier(random_state=self.random_state, **self.parm_dic)

            clf.fit(X_train, Y_train)
            self.clfs.append(clf)
    def predict_proba(self,X_test):
        X_test_drup = X_test[:, self.choice_feature_drug]
        X_test_tp = X_test[:, self.choice_feature_tp]
        if self.reduce_dim_ratio<1:
            X_test_drup = self.drug_svd.transform(X_test_drup)
            X_test_tp = self.tp_svd.transform(X_test_tp)
        X_test = np.concatenate([X_test_drup, X_test_tp], axis=1)
        preds = []
        for i in range(self.k):
            pred = (self.clfs[i].predict_proba(X_test))[:, 1].flatten()
            # pred = (self.clfs[i].predict(X_test[:, self.choice_features]))
            preds.append(list(pred))
        preds = np.array(preds)
        preds = np.mean(preds, axis=0)
        return preds
    def predict_proba_single_learner(self,X_test,index):
        X_test_drup = X_test[:, self.choice_feature_drug]
        X_test_tp = X_test[:, self.choice_feature_tp]
        if self.reduce_dim_ratio<1:
            X_test_drup = self.drug_svd.transform(X_test_drup)
            X_test_tp = self.tp_svd.transform(X_test_tp)
        X_test = np.concatenate([X_test_drup, X_test_tp], axis=1)

        pred = (self.clfs[index].predict_proba(X_test))[:, 1].flatten()
        # pred = (self.clfs[i].predict(X_test[:, self.choice_features]))
        return pred
    def predict(self,X_test):
        preds = self.predict_proba(X_test)
        pred_label = np.zeros(preds.shape[0])
        pred_label[preds >= 0.5] = 1
        return pred_label
    pass

