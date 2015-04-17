# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import datasets, cluster, decomposition, linear_model
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

class PipelineSample(object):
    def plot_pipeline(self):
        logistic = linear_model.LogisticRegression()
        pca = decomposition.PCA()
        # 主成分分析の結果(次元削除)をロジスティック回帰でデータあてはめ
        pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
        
        digits = datasets.load_digits()
        X_digits = digits.data
        y_digits = digits.target
        
        ###############################################################################
        # Plot the PCA spectrum
        pca.fit(X_digits)
        
        plt.figure(1, figsize=(4, 3))
        plt.clf()
        plt.axes([.2, .2, .7, .7])
        plt.plot(pca.explained_variance_, linewidth=2)
        plt.axis('tight')
        plt.xlabel('n_components')
        plt.ylabel('explained_variance')
        
        ###############################################################################
        # Prediction
        
        n_components = [20, 40, 64]
        Cs = np.logspace(-4, 4, 3)
        
        #Parameters of pipelines can be set using ‘__’ separated parameter names:
        # 主成分分析の次元数とロジスティック回帰分析の誤り許容度で最適なパラメータ探索
        estimator = GridSearchCV(pipe,
                                 dict(pca__n_components=n_components,
                                      logistic__C=Cs))
        estimator.fit(X_digits, y_digits)
        
        plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
                    linestyle=':', label='n_components chosen')
        plt.legend(prop=dict(size=12))
        plt.savefig("pipline.jpg")

if __name__ == '__main__':
    obj = PipelineSample()
    obj.plot_pipeline()