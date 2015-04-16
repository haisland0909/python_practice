# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.decomposition import PCA
class KNearestSample(object):
    def __init__(self):
        self._iris = datasets.load_iris()

    def get_iris_x(self):

        return self._iris.data

    def get_iris_y(self):

        return self._iris.target

    def plot_iris_data(self):
        fig       = plt.figure(1, figsize=(8, 6))
        ax        = Axes3D(fig, elev=-150, azim=110)
        X_reduced = PCA(n_components = 3).fit_transform(self.get_iris_x())

        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c = self.get_iris_y(),
                   cmap=plt.cm.Paired)
        ax.set_title("First three PCA directions")
        ax.set_xlabel("1st eigenvector")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd eigenvector")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd eigenvector")
        ax.w_zaxis.set_ticklabels([])
        plt.savefig("iris_data_3d.jpg")

    def do_k_nearest(self, neighbor = 5):
        iris_X = self.get_iris_x()[:, :2]
        iris_y = self.get_iris_y()
        np.random.seed(0)
        indices = np.random.permutation(len(iris_X))
        iris_X_train = iris_X[indices[:-10]]
        iris_y_train = iris_y[indices[:-10]]
        iris_X_test  = iris_X[indices[-10:]]
        iris_y_test  = iris_y[indices[-10:]]
        # 2次元で表記する
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
        cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        plt.pcolormesh(xx, yy, Z, cmap = cmap_light)

        # Plot also the training points
        plt.scatter(iris_X_test[:, 0], iris_X_test[:, 1], c = iris_y_test, cmap = cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
              % (neighbor, "uniform"))
        plt.savefig("k-nearest_" + str(neighbor) + ".jpg")




class LinearSample(object):
    def __init__(self):
        self._diabetes = datasets.load_diabetes()
        self._shrink_x = np.c_[ .5, 1].T
        self._shrink_y = [.5, 1]
        self._shrink_t = np.c_[ 0, 2].T

    def get_diabetes_x(self):

        return self._diabetes.data

    def get_diabetes_y(self):

        return self._diabetes.target

    def do_linear_regression(self):
        diabetes_X      = self.get_diabetes_x()[:, np.newaxis]
        diabetes_X_temp = diabetes_X[:, :, 2]
        diabetes_X_train = diabetes_X_temp[:-20]
        diabetes_y_train = self.get_diabetes_y()[:-20]
        diabetes_X_test  = diabetes_X_temp[-20:]
        diabetes_y_test  = self.get_diabetes_y()[-20:]
        regr = linear_model.LinearRegression()
        regr.fit(diabetes_X_train, diabetes_y_train)
        plt.figure()
        plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
        plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
                 linewidth=3)
        plt.xticks(())
        plt.yticks(())
        plt.title("Linear Regression")

        plt.savefig("linear_regression.png")

    def do_linear_few_data(self, fit_way = linear_model.LinearRegression()):
        plt.figure() 
        np.random.seed(0)
        for _ in range(6): 
            this_X = .1*np.random.normal(size=(2, 1)) + self._shrink_x
            fit_way.fit(this_X, self._shrink_y)
            plt.plot(self._shrink_t, fit_way.predict(self._shrink_t)) 
            plt.scatter(this_X, self._shrink_y, s=3)  
        plt.title("Faw Data Linear Regression BY %s" % fit_way.__class__.__name__)
        plt.savefig("linear_few_%s.jpg" % fit_way.__class__.__name__)

if __name__ == '__main__':
    obj = KNearestSample()
    obj.plot_iris_data()
    obj.do_k_nearest(1)
    obj.do_k_nearest(5)
    obj.do_k_nearest(10)
    
    obj = LinearSample()
    obj.do_linear_regression()

