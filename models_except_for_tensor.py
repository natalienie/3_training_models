import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

def train_test_split(feature, label, test_potion=0.2):
    '''
    feature is the df data
    lable is a pd.Series
    this function will return  X_train, y_train, X_test, y_test
    in the form of (None, 2), (None,1) array
    '''
    leng = len(feature)
    leng_test = int(test_potion * leng)
    X_train = feature[:-1 * leng_test]
    y_train = label[:-1 * leng_test]
    X_test = feature[-1 * leng_test:]
    y_test = label[-1 * leng_test:]


    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train).reshape(-1,1)
    y_test = np.array(y_test).reshape(-1,1)

    return X_train, y_train, X_test, y_test

def Apply_RandomForest(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(max_depth=3, random_state=0)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    accuracy = clf.score(prediction, y_test)
    price_importance, macd_importance = cfl.feature_importances_
    return accuracy

def Apply_SVM(X_train, y_train, X_test, y_test):
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    accuracy = clf.score(prediction, y_test)
    return accuracy

def Apply_GaussianNB(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    predition = model.predict(X_test)
    accuracy = model.score(prediction, y_test)
    return accuracy

def find_best_model(X_train, y_train, X_test, y_test):
    SVM_accuracy = Apply_SVM(X_train, y_train, X_test, y_test)
    print('Accuracy of SVM model is: {}'.format(SVM_accuracy))
    Gaussian_accuracy = Apply_GaussianNB(X_train, y_train, X_test, y_test)
    print('Accuracy of Gaussian model is: {}'.format(Gaussian_accuracy))
    Forest_accuracy = Apply_RandomForest(X_train, y_train, X_test, y_test)
    print('Accuracy of RandomForest model is: {}'.format(Forest_accuracy))
