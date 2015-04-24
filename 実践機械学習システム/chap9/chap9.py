# -*- coding: utf-8 -*-
# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, auc
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model.logistic import LogisticRegression
from utils import plot_pr, plot_roc, plot_confusion_matrix, GENRE_LIST
from fft  import read_fft
from ceps import read_ceps
from abc  import ABCMeta, abstractmethod

class AbstractSoundClassifyBase(object):
    __metaclass__ = ABCMeta
    genre_list    = GENRE_LIST

    def __init__(self, clf_model = LogisticRegression()):
        self._clf_model  = clf_model
        self._X = None 
        self._Y = None

    @abstractmethod
    def train_model(self):
        print "abstract"

    @abstractmethod
    def get_way_name(self):
        return "abstract", "abstract way"

    def create_model(self):
        
        clf = LogisticRegression()

        return clf

class FFTSoundClassify(AbstractSoundClassifyBase):
    def __init__(self):
        super(FFTSoundClassify, self).__init__()
        self._X, self._Y = read_fft(GENRE_LIST)

    def train_model(self, name, plot=False):
        X          = self._X
        Y          = self._Y
        labels     = np.unique(Y)
        genre_list = AbstractSoundClassifyBase.genre_list
        cv = ShuffleSplit(
            n=len(X), n_iter=1, test_size=0.3, indices=True, random_state=0)

        train_errors = []
        test_errors  = []

        scores    = []
        pr_scores = defaultdict(list)
        precisions, recalls, thresholds = defaultdict(
            list), defaultdict(list), defaultdict(list)

        roc_scores = defaultdict(list)
        tprs       = defaultdict(list)
        fprs       = defaultdict(list)

        clfs = []  # just to later get the median

        cms  = []

        for train, test in cv:
            X_train, y_train = X[train], Y[train]
            X_test, y_test = X[test], Y[test]

            clf = self._clf_model
            clf.fit(X_train, y_train)
            clfs.append(clf)

            train_score = clf.score(X_train, y_train)
            test_score = clf.score(X_test, y_test)
            scores.append(test_score)

            train_errors.append(1 - train_score)
            test_errors.append(1 - test_score)

            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            cms.append(cm)

            for label in labels:
                y_label_test = np.asarray(y_test == label, dtype=int)
                proba = clf.predict_proba(X_test)
                proba_label = proba[:, label]

                precision, recall, pr_thresholds = precision_recall_curve(
                    y_label_test, proba_label)
                pr_scores[label].append(auc(recall, precision))
                precisions[label].append(precision)
                recalls[label].append(recall)
                thresholds[label].append(pr_thresholds)

                fpr, tpr, roc_thresholds = roc_curve(y_label_test, proba_label)
                roc_scores[label].append(auc(fpr, tpr))
                tprs[label].append(tpr)
                fprs[label].append(fpr)

        if plot:
            for label in labels:
                print("Plotting %s" % genre_list[label])
                scores_to_sort = roc_scores[label]
                median = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]

                desc = "%s %s" % (name, genre_list[label])
                plot_pr(pr_scores[label][median], desc, precisions[label][median],
                        recalls[label][median], label='%s vs rest' % genre_list[label])
                plot_roc(roc_scores[label][median], desc, tprs[label][median],
                         fprs[label][median], label='%s vs rest' % genre_list[label])

        all_pr_scores = np.asarray(pr_scores.values()).flatten()
        summary = (np.mean(scores), np.std(scores),
                   np.mean(all_pr_scores), np.std(all_pr_scores))
        print("%.3f\t%.3f\t%.3f\t%.3f\t" % summary)

        return np.mean(train_errors), np.mean(test_errors), np.asarray(cms)

    def get_way_name(self):
        return "fft", "Confusion matrix of an FFT based classifier"

class CepsSoundClassify(AbstractSoundClassifyBase):
    def __init__(self):
        super(CepsSoundClassify, self).__init__()
        self._X, self._Y = read_ceps(GENRE_LIST)

    def train_model(self, name, plot=False):
        X      = self._X
        Y      = self._Y
        labels = np.unique(Y)
        genre_list = AbstractSoundClassifyBase.genre_list
        cv = ShuffleSplit(
            n=len(X), n_iter=1, test_size=0.3, indices=True, random_state=0)

        train_errors = []
        test_errors = []

        scores = []
        pr_scores = defaultdict(list)
        precisions, recalls, thresholds = defaultdict(
            list), defaultdict(list), defaultdict(list)

        roc_scores = defaultdict(list)
        tprs = defaultdict(list)
        fprs = defaultdict(list)

        clfs = []  # just to later get the median

        cms = []

        for train, test in cv:
            X_train, y_train = X[train], Y[train]
            X_test, y_test = X[test], Y[test]

            clf = self._clf_model
            clf.fit(X_train, y_train)
            clfs.append(clf)

            train_score = clf.score(X_train, y_train)
            test_score = clf.score(X_test, y_test)
            scores.append(test_score)

            train_errors.append(1 - train_score)
            test_errors.append(1 - test_score)

            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            cms.append(cm)

            for label in labels:
                y_label_test = np.asarray(y_test == label, dtype=int)
                proba = clf.predict_proba(X_test)
                proba_label = proba[:, label]

                precision, recall, pr_thresholds = precision_recall_curve(
                    y_label_test, proba_label)
                pr_scores[label].append(auc(recall, precision))
                precisions[label].append(precision)
                recalls[label].append(recall)
                thresholds[label].append(pr_thresholds)

                fpr, tpr, roc_thresholds = roc_curve(y_label_test, proba_label)
                roc_scores[label].append(auc(fpr, tpr))
                tprs[label].append(tpr)
                fprs[label].append(fpr)

        if plot:
            for label in labels:
                print("Plotting %s" % genre_list[label])
                scores_to_sort = roc_scores[label]
                median = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]

                desc = "%s %s" % (name, genre_list[label])
                plot_roc(roc_scores[label][median], desc, tprs[label][median],
                         fprs[label][median], label='%s vs rest' % genre_list[label])

        all_pr_scores = np.asarray(pr_scores.values()).flatten()
        summary = (np.mean(scores), np.std(scores),
                   np.mean(all_pr_scores), np.std(all_pr_scores))
        print("%.3f\t%.3f\t%.3f\t%.3f\t" % summary)

        return np.mean(train_errors), np.mean(test_errors), np.asarray(cms)

    def get_way_name(self):
        return "ceps", "Confusion matrix of an Ceps based classifier"



class Chap9(object):
    def __init__(self, classify_obj):
        self._classify_obj = classify_obj

    def do_classify(self, name):
        train_avg, test_avg, cms = self._classify_obj.train_model(name, plot=True)
        cm_avg = np.mean(cms, axis=0)
        cm_norm = cm_avg / np.sum(cm_avg, axis=0)
        name, desc = self._classify_obj.get_way_name()
        plot_confusion_matrix(cm_norm, AbstractSoundClassifyBase.genre_list, name, desc)
    



if __name__ == "__main__":
    obj = Chap9(FFTSoundClassify())
    obj.do_classify("Log Reg FFT")
    obj = Chap9(CepsSoundClassify())
    obj.do_classify("Log Reg Ceps")
    