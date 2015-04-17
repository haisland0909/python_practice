# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, svm
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
        self._alphas   = np.logspace(-4, -1, 6)

    def get_diabetes_x(self):

        return self._diabetes.data

    def get_diabetes_y(self):

        return self._diabetes.target
    

    def do_linear_regression(self):
        diabetes_X       = self.get_diabetes_x()[:, np.newaxis]
        diabetes_X_temp  = diabetes_X[:, :, 2]
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

    def plot_sparce_data(self):
        diabetes = self.get_diabetes_x()
        indices  = (0, 1)
        X_train  = self.get_diabetes_x()[:-20, indices]
        X_test   = self.get_diabetes_x()[-20:, indices]
        y_train  = self.get_diabetes_y()[:-20]
        y_test   = self.get_diabetes_y()[-20:]
        ols = linear_model.LinearRegression()
        ols.fit(X_train, y_train)
        elev = 43.5
        azim = -110
        self._plot_figs(1, elev, azim, X_train, y_train, ols)
        
        elev = -.5
        azim = 0
        self._plot_figs(2, elev, azim, X_train, y_train, ols)
        
        elev = -.5
        azim = 90
        self._plot_figs(3, elev, azim, X_train, y_train, ols)

    def _plot_figs(self, fig_num, elev, azim, X_train, y_train, clf):
        fig = plt.figure(fig_num, figsize=(4, 3))
        plt.clf()
        ax = Axes3D(fig, elev=elev, azim=azim)

        ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='k', marker='+')
        ax.plot_surface(np.array([[-.1, -.1], [.15, .15]]),
                        np.array([[-.1, .15], [-.1, .15]]),
                        clf.predict(np.array([[-.1, -.1, .15, .15],
                                              [-.1, .15, -.1, .15]]).T
                                    ).reshape((2, 2)),
                        alpha=.5)
        ax.set_xlabel('X_1')
        ax.set_ylabel('X_2')
        ax.set_zlabel('Y')
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        plt.savefig("sparce_data_%d.jpg" % fig_num)

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

    def do_lasso(self):
        diabetes_X       = self.get_diabetes_x()
        diabetes_X_train = diabetes_X[:-20]
        diabetes_y_train = self.get_diabetes_y()[:-20]
        diabetes_X_test  = diabetes_X[-20:]
        diabetes_y_test  = self.get_diabetes_y()[-20:]
        regr             = linear_model.Lasso()
        scores           = [regr.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train).score(diabetes_X_test, diabetes_y_test) for alpha in self._alphas]
        best_alpha       = self._alphas[scores.index(max(scores))]
        regr.alpha       = best_alpha
        regr.fit(diabetes_X_train, diabetes_y_train)
        print(regr.coef_)

    def model(self, x):

        return 1 / (1 + np.exp(-x))

    def plot_logistics(self):
        xmin, xmax = -5, 5
        n_samples  = 100
        np.random.seed(0)
        X          = np.random.normal(size=n_samples)
        y          = (X > 0).astype(np.float)
        X[X > 0]   *= 4
        X          += .3 * np.random.normal(size=n_samples)       
        X          = X[:, np.newaxis]
        # run the classifier
        clf        = linear_model.LogisticRegression(C=1e5)
        clf.fit(X, y)
        
        # and plot the result
        plt.figure(1, figsize=(4, 3))
        plt.clf()
        plt.scatter(X.ravel(), y, color='black', zorder=20)
        X_test = np.linspace(-5, 10, 300)
        loss   = self.model(X_test * clf.coef_ + clf.intercept_).ravel()
        plt.plot(X_test, loss, color='blue', linewidth=3)
        
        ols    = linear_model.LinearRegression()
        ols.fit(X, y)
        plt.plot(X_test, ols.coef_ * X_test + ols.intercept_, linewidth=1)
        plt.axhline(.5, color='.5')
        plt.ylabel('y')
        plt.xlabel('X')
        plt.xticks(())
        plt.yticks(())
        plt.ylim(-.25, 1.25)
        plt.xlim(-4, 10)
        plt.savefig("logistics_regression.jpg")

class SVMSample(object):
    def __init__(self):
        self._iris = datasets.load_iris()

    def get_iris_x(self):

        return self._iris.data

    def get_iris_y(self):

        return self._iris.target

    def plot_svm_margin(self):
        np.random.seed(0)
        X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
        Y = [0] * 20 + [1] * 20
        
        # figure number
        fignum = 1
        
        # fit the model
        for name, penalty in (('linear', 1), ('poly', 0.05), ('rbf', 1)):
        
            clf = svm.SVC(kernel=name, C=penalty)
            clf.fit(X, Y)
        
            # plot the line, the points, and the nearest vectors to the plane
            plt.figure(fignum, figsize=(4, 3))
            plt.clf()
            # サポートベクターには大きめの丸を重ねて描く
            plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                        facecolors='none', zorder=10)
            plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)
        
            plt.axis('tight')
            x_min = -4.8
            x_max = 4.2
            y_min = -6
            y_max = 6
        
            XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
            Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
        
            # Put the result into a color plot
            Z = Z.reshape(XX.shape)
            plt.figure(fignum, figsize=(4, 3))
            plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
            plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])
        
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
        
            plt.xticks(())
            plt.yticks(())
            plt.savefig("svm_sample_%s.jpg" % name)
            fignum = fignum + 1


        
if __name__ == '__main__':
    obj = KNearestSample()
    obj.plot_iris_data()
    obj.do_k_nearest(1)
    obj.do_k_nearest(5)
    obj.do_k_nearest(10)
    
    obj = LinearSample()
    obj.do_linear_regression()

