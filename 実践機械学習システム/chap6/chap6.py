# -*- coding: utf-8 -*-
# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

#
# This script trains multinomial Naive Bayes on the tweet corpus
# to find two different results:
# - How well can we distinguis positive from negative tweets?
# - How well can we detect whether a tweet contains sentiment at all?
#

import time
start_time = time.time()

import numpy as np
import re

from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.cross_validation import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils import plot_pr
from utils import load_sanders_data
from utils import tweak_labels
from utils import log_false_positives
from utils import load_sent_word_net
class Chap6(object):

    def __init__(self):
        self._pipeline_model = None
        self._X              = None
        self._Y              = None
        self._emo_repl       = {
            # positive emoticons
            "&lt;3": " good ",
            ":d": " good ",  # :D in lower case
            ":dd": " good ",  # :DD in lower case
            "8)": " good ",
            ":-)": " good ",
            ":)": " good ",
            ";)": " good ",
            "(-:": " good ",
            "(:": " good ",
    
            # negative emoticons:
            ":/": " bad ",
            ":&gt;": " sad ",
            ":')": " sad ",
            ":-(": " bad ",
            ":(": " bad ",
            ":S": " bad ",
            ":-S": " bad ",
        }
        self._emo_repl_order = [k for (k_len, k) in reversed(sorted([(len(k), k) for k in list(self._emo_repl.keys())]))]
        self._re_repl        = {
            r"\br\b": "are",
            r"\bu\b": "you",
            r"\bhaha\b": "ha",
            r"\bhahaha\b": "ha",
            r"\bdon't\b": "do not",
            r"\bdoesn't\b": "does not",
            r"\bdidn't\b": "did not",
            r"\bhasn't\b": "has not",
            r"\bhaven't\b": "have not",
            r"\bhadn't\b": "had not",
            r"\bwon't\b": "will not",
            r"\bwouldn't\b": "would not",
            r"\bcan't\b": "can not",
            r"\bcannot\b": "can not",
        }


    def set_X(self, X):
        self._X = X

    def set_Y(self, Y):
        self._Y = Y

    def create_ngram_model(self):
        tfidf_ngrams = TfidfVectorizer(ngram_range=(1, 3), analyzer="word", binary=False)
        clf          = MultinomialNB()
        pipeline     = Pipeline([('vect', tfidf_ngrams), ('clf', clf)])

        return pipeline

    def create_ngram_model_with_emo(self, params=None):
        def preprocessor(tweet):
            global emoticons_replaced
            tweet = tweet.lower()

            for k in self._emo_repl_order:
                tweet = tweet.replace(k, self._emo_repl[k])
            for r, repl in self._re_repl.items():
                tweet = re.sub(r, repl, tweet)

            return tweet

        tfidf_ngrams = TfidfVectorizer(preprocessor=preprocessor,
                                       analyzer="word")
        clf = MultinomialNB()
        pipeline = Pipeline([('tfidf', tfidf_ngrams), ('clf', clf)])

        if params:
            pipeline.set_params(**params)

        return pipeline

    def get_best_model(self):
        best_params = dict(tfidf__ngram_range=(1, 2),
                           tfidf__min_df=1,
                           tfidf__stop_words=None,
                           tfidf__smooth_idf=False,
                           tfidf__use_idf=False,
                           tfidf__sublinear_tf=True,
                           tfidf__binary=False,
                           clf__alpha=0.01,
                           )

        best_clf = self.create_ngram_model_with_emo(best_params)

        return best_clf

    def train_model(self, clf_factory, name="NB ngram", plot=False):
        X            = self._X
        Y            = self._Y
        cv           = ShuffleSplit(n=len(X), n_iter=10, test_size=0.3, random_state=0)
        train_errors = []
        test_errors  = []
        scores       = []
        pr_scores    = []
        precisions, recalls, thresholds = [], [], []
        
        for train, test in cv:
            X_train, y_train = X[train], Y[train]
            X_test, y_test = X[test], Y[test]

            clf = clf_factory()
            clf.fit(X_train, y_train)

            train_score = clf.score(X_train, y_train)
            test_score = clf.score(X_test, y_test)

            train_errors.append(1 - train_score)
            test_errors.append(1 - test_score)

            scores.append(test_score)
            proba = clf.predict_proba(X_test)

            fpr, tpr, roc_thresholds = roc_curve(y_test, proba[:, 1])
            precision, recall, pr_thresholds = precision_recall_curve(
                y_test, proba[:, 1])

            pr_scores.append(auc(recall, precision))
            precisions.append(precision)
            recalls.append(recall)
            thresholds.append(pr_thresholds)

        scores_to_sort = pr_scores
        median = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]

        if plot:
            plot_pr(pr_scores[median], name, "01", precisions[median],
                    recalls[median], label=name)

            summary = (np.mean(scores), np.std(scores),
                       np.mean(pr_scores), np.std(pr_scores))
            print("%.3f\t%.3f\t%.3f\t%.3f\t" % summary)

        return np.mean(train_errors), np.mean(test_errors)

    def print_incorrect(self, clf, X, Y):
        Y_hat       = clf.predict(X)
        wrong_idx   = Y_hat != Y
        X_wrong     = X[wrong_idx]
        Y_wrong     = Y[wrong_idx]
        Y_hat_wrong = Y_hat[wrong_idx]
        for idx in range(len(X_wrong)):
            print("clf.predict('%s')=%i instead of %i" %
                  (X_wrong[idx], Y_hat_wrong[idx], Y_wrong[idx]))

if __name__ == "__main__":
    obj = Chap6()
    X_orig, Y_orig = load_sanders_data()
    classes = np.unique(Y_orig)
    for c in classes:
        print("#%s: %i" % (c, sum(Y_orig == c)))

    print("== Pos vs. neg ==")
    pos_neg = np.logical_or(Y_orig == "positive", Y_orig == "negative")
    X = X_orig[pos_neg]
    Y = Y_orig[pos_neg]
    Y = tweak_labels(Y, ["positive"])
    obj.set_X(X)
    obj.set_Y(Y)
    obj.train_model(obj.create_ngram_model, name="pos vs neg", plot=True)
    obj.train_model(obj.get_best_model, name="pos vs neg best", plot=True)

    print("== Pos/neg vs. irrelevant/neutral ==")
    X = X_orig
    Y = tweak_labels(Y_orig, ["positive", "negative"])
    obj.set_X(X)
    obj.set_Y(Y)
    obj.train_model(obj.create_ngram_model, name="sent vs rest", plot=True)
    obj.train_model(obj.get_best_model, name="sent vs rest best", plot=True)

    print("== Pos vs. rest ==")
    X = X_orig
    Y = tweak_labels(Y_orig, ["positive"])
    obj.set_X(X)
    obj.set_Y(Y)
    obj.train_model(obj.create_ngram_model, name="pos vs rest", plot=True)
    obj.train_model(obj.get_best_model, name="pos vs rest best", plot=True)

    print("== Neg vs. rest ==")
    X = X_orig
    Y = tweak_labels(Y_orig, ["negative"])
    obj.set_X(X)
    obj.set_Y(Y)
    obj.train_model(obj.create_ngram_model, name="neg vs rest", plot=True)
    obj.train_model(obj.get_best_model, name="neg vs rest best", plot=True)

 

    print("time spent:", time.time() - start_time)