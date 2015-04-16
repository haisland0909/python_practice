# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.colors import ListedColormap
class SuperVisedLearningSample(object):
    def __init__(self):
        self._iris = datasets.load_iris()

    def get_iris_x(self):

        return self._iris.data

    def get_iris_y(self):

        return self._iris.target

    def do_k_nearest(self, neighbor = 5):
        iris_X = self.get_iris_x()[:, :2]
        iris_y = self.get_iris_y()
        np.random.seed(0)
        indices = np.random.permutation(len(iris_X))
        iris_X_train = iris_X[indices[:-10]]
        iris_y_train = iris_y[indices[:-10]]
        iris_X_test  = iris_X[indices[-10:]]
        iris_y_test  = iris_y[indices[-10:]]
        # 2ŽŸŒ³‚Å•\‹L‚·‚é
        h = .02
        x_min, x_max = iris_X_test[:, 0].min() - 1, iris_X_test[:, 0].max() + 1
        y_min, y_max = iris_X_test[:, 1].min() - 1, iris_X_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors = neighbor)
        knn.fit(iris_X_train, iris_y_train) 
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure()
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        plt.pcolormesh(xx, yy, Z, cmap = cmap_light)

        # Plot also the training points
        plt.scatter(iris_X_test[:, 0], iris_X_test[:, 1], c = iris_y_test, cmap = cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
              % (neighbor, "uniform"))
        plt.savefig("k-nearest_" + str(neighbor) + ".jpg")


obj = SuperVisedLearningSample()
obj.do_k_nearest(1)
obj.do_k_nearest(5)
obj.do_k_nearest(10)


