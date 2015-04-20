# -*- coding: utf-8 -*-
# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import sklearn.datasets
import scipy as sp
import nltk.stem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics

class Chap3Sample(object):
    def __init__(self):
        self._new_post        = \
        """Disk drive problems. Hi, I have a problem with my hard disk.
        After 1 year it is working only sporadically now.
        I tried to format it, but now it doesn't boot any more.
        Any ideas? Thanks.
        """
        self._all_data        = sklearn.datasets.fetch_20newsgroups(subset="all")

        groups                = [
            'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
            'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space'
        ]

        self._train_data      = sklearn.datasets.fetch_20newsgroups(subset="train", categories=groups)
        self._vectorized_data = None
        self._data_label      = self._train_data.target
        self._result_label    = None
        self._kmean_solver    = None

    def vectorize(self):
        self._vectorizer      = StemmedTfidfVectorizer(min_df=10, max_df=0.5, stop_words='english', decode_error='ignore')
        self._vectorized_data = self._vectorizer.fit_transform(self._train_data.data)

    def get_vectorized_data(self):

        return self._vectorized_data

    def get_vector_shape(self):

        return self._vectorized_data.shape()

    def set_kmean_solver(self, num_clusters = 50):
        self._kmean_solver = KMeans(n_clusters=num_clusters, n_init=1, verbose=1, random_state=3)

    def do_kmean(self):
        if self._vectorized_data is None:
            self.vectorize()

        if self._kmean_solver is None:
            self.set_kmean_solver()
        self._kmean_solver.fit(self._vectorized_data)
        self._result_label = self._kmean_solver.labels_

    def get_homo_score(self):

        return  metrics.homogeneity_score(self._data_label, self._result_label)

    def get_comp_score(self):

        return  metrics.completeness_score(self._data_label, self._result_label)

    def get_vmeasure_score(self):

        return metrics.v_measure_score(self._data_label, self._result_label)

    def get_similar_to_new_post(self):
        if self._result_label is None:
            self.do_kmean()
        new_post_vector  = self._vectorizer.transform([self._new_post])
        new_post_label   = self._kmean_solver.predict(new_post_vector)[0]
        # 同一クラスタに存在するもののみ
        similar_indices  = (self._result_label == new_post_label).nonzero()[0]
        similar          = []
        for i in similar_indices:
            dist = sp.linalg.norm((new_post_vector - self._vectorized_data[i]).toarray())
            similar.append((dist, self._train_data.data[i]))       
        similar = sorted(similar)

        return similar




class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (nltk.stem.SnowballStemmer('english').stem(w) for w in analyzer(doc))



