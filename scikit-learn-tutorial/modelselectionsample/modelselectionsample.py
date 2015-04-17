# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets, linear_model, svm

class CrossValidationSample(object):
    def __init__(self):
        digits         = datasets.load_digits()
        self._x_digits = digits.data
        self._y_digits = digits.target
        self.fit_way   = None

    def set_fit_way(self, way = svm.SVC(C=1, kernel='linear')):
        self.fit_way = way

    def hold_out_validation(self, test_num = 100):
        if self.fit_way is None:
            self.set_fit_way()

        return self.fit_way.fit(self._x_digits[:-test_num], self._y_digits[:-test_num]).score(self._x_digits[-test_num:], self._y_digits[-test_num:])

    
    def k_fold_validation(self, k = 3):
        if self.fit_way is None:
            self.set_fit_way()
        X_folds = np.array_split(self._x_digits, k)
        y_folds = np.array_split(self._y_digits, k)
        scores  = list()
        for i in range(k):
            X_train = list(X_folds)
            X_test  = X_train.pop(i)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test  = y_train.pop(i)
            y_train = np.concatenate(y_train)
            scores.append(self.fit_way.fit(X_train, y_train).score(X_test, y_test))

        return scores