# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import datasets, cluster
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
class KMeansSample(object):
    def __init__(self):
        iris         = datasets.load_iris()
        self._x_iris = iris.data
        self._y_iris = iris.target
        try:
           self._lena = sp.lena()
        except AttributeError:
           from scipy import misc
           self._lena = misc.lena()

    def plot_iris_k_mean(self):
        X          = self._x_iris
        y          = self._y_iris
        estimators = {
              'k_means_iris_3' : cluster.KMeans(n_clusters = 3),
              'k_means_iris_8' : cluster.KMeans(n_clusters = 8),
              'k_means_iris_bad_init' : cluster.KMeans(n_clusters = 3, n_init = 1, init='random')
        }
        fignum     = 1
        for name, est in estimators.items():
            fig = plt.figure(fignum, figsize=(4, 3))
            plt.clf()
            ax  = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        
            plt.cla()
            est.fit(X)
            labels = est.labels_
        
            ax.scatter(X[:, 3], X[:, 0], X[:, 2], c = labels.astype(np.float))
        
            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
            ax.set_xlabel('Petal width')
            ax.set_ylabel('Sepal length')
            ax.set_zlabel('Petal length')
            plt.savefig("k-means-iris-%s.jpg" % name)
            fignum = fignum + 1
        
        # Plot the ground truth
        fig = plt.figure(fignum, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        
        plt.cla()
        
        for name, label in [('Setosa', 0),
                            ('Versicolour', 1),
                            ('Virginica', 2)]:
            ax.text3D(X[y == label, 3].mean(),
                      X[y == label, 0].mean() + 1.5,
                      X[y == label, 2].mean(), name,
                      horizontalalignment='center',
                      bbox=dict(alpha = .5, edgecolor = 'w', facecolor = 'w'))
        # Reorder the labels to have colors matching the cluster results
        y = np.choose(y, [1, 2, 0]).astype(np.float)
        ax.scatter(X[:, 3], X[:, 0], X[:, 2], c = y)
        
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')
        plt.savefig("k-means-iris-truth.jpg")

    def plot_lena_k_mean(self):
        n_clusters = 5
        np.random.seed(0)
        lena       = self._lena
        X          = lena.reshape((-1, 1))  # We need an (n_sample, n_feature) array
        k_means    = cluster.KMeans(n_clusters=n_clusters, n_init=4)
        k_means.fit(X)
        values     = k_means.cluster_centers_.squeeze()
        labels     = k_means.labels_
        
        # create an array from labels and values
        lena_compressed       = np.choose(labels, values)
        lena_compressed.shape = lena.shape
        
        vmin       = lena.min()
        vmax       = lena.max()
        
        # original lena
        plt.figure(1, figsize=(3, 2.2))
        plt.imshow(lena, cmap=plt.cm.gray, vmin=vmin, vmax=256)
        plt.savefig("lena_original.jpg")
        
        # compressed lena
        plt.figure(2, figsize=(3, 2.2))
        plt.imshow(lena_compressed, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        plt.savefig("lena_k_means.jpg")
        # equal bins lena
        regular_values = np.linspace(0, 256, n_clusters + 1)
        
        regular_labels = np.searchsorted(regular_values, lena) - 1
        regular_values = .5 * (regular_values[1:] + regular_values[:-1])  # mean
        regular_lena = np.choose(regular_labels.ravel(), regular_values)
        regular_lena.shape = lena.shape
        
        plt.figure(3, figsize=(3, 2.2))
        plt.imshow(regular_lena, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        plt.savefig("lena_regular.jpg")
        
        # histogram
        plt.figure(4, figsize=(3, 2.2))
        plt.clf()
        plt.axes([.01, .01, .98, .98])
        plt.hist(X, bins=256, color='.5', edgecolor='.5')
        plt.yticks(())
        plt.xticks(regular_values)
        values = np.sort(values)
        for center_1, center_2 in zip(values[:-1], values[1:]):
            plt.axvline(.5 * (center_1 + center_2), color='b')
        
        for center_1, center_2 in zip(regular_values[:-1], regular_values[1:]):
            plt.axvline(.5 * (center_1 + center_2), color='b', linestyle='--')

        plt.savefig("lena_hist.jpg")

    def plot_ward_cluster_lena(self):
        lena = self._lena
        # Downsample the image by a factor of 4
        lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
        X    = np.reshape(lena, (-1, 1))
        # Define the structure A of the data. Pixels connected to their neighbors.
        connectivity = grid_to_graph(*lena.shape)
        # Compute clustering
        print("Compute structured hierarchical clustering...")
        n_clusters = 15  # number of regions
        ward = AgglomerativeClustering(n_clusters=n_clusters,
                linkage='ward', connectivity=connectivity).fit(X)
        label = np.reshape(ward.labels_, lena.shape)
        plt.figure()
        plt.imshow(lena, cmap=plt.cm.gray)
        for l in range(n_clusters):
            plt.contour(label == l, contours=1,
                        colors=[plt.cm.spectral(l / float(n_clusters)), ])
        plt.xticks(())
        plt.yticks(())
        plt.savefig("lena_ward_cluster.jpg")
