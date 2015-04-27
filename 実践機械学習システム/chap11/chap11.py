# -*- coding: utf-8 -*-
# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import os

from matplotlib import pylab
import numpy as np
import scipy
from scipy.stats import norm, pearsonr

from utils import CHART_DIR

class Correlation(object):
    def __init__(self):
        self._X = None
        self._Y = None

    def set_data_set(self, x, y):
        self._X = x
        self._Y = y


    def plot_correlation_func(self, fig_num):
        x    = self._X
        y    = self._Y
        r, p = pearsonr(x, y)
        title = "Cor($X_1$, $X_2$) = %.3f" % r
        pylab.subplot(fig_num)
        pylab.scatter(x, y)
        pylab.title(title)
        pylab.xlabel("$X_1$")
        pylab.ylabel("$X_2$")

        f1 = scipy.poly1d(scipy.polyfit(x, y, 1))
        pylab.plot(x, f1(x), "r--", linewidth=2)


if __name__ == "__main__":
    obj = Correlation()
    np.random.seed(0)  # to reproduce the data later on
    pylab.clf()
    pylab.figure(num=None, figsize=(8, 8))
    x = np.arange(0, 10, 0.2)
    y = 0.5 * x + norm.rvs(1, scale=.01, size=len(x))
    obj.set_data_set(x, y)
    obj.plot_correlation_func(221)
    y = 0.5 * x + norm.rvs(1, scale=.1, size=len(x))
    obj.set_data_set(x, y)
    obj.plot_correlation_func(222)
    y = 0.5 * x + norm.rvs(1, scale=1, size=len(x))
    obj.set_data_set(x, y)
    obj.plot_correlation_func(223)
    y = norm.rvs(1, scale=10, size=len(x))
    obj.set_data_set(x, y)
    obj.plot_correlation_func(224)
    pylab.autoscale(tight=True)
    pylab.grid(True)
    filename = "corr_demo_1.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")

    pylab.clf()
    pylab.figure(num=None, figsize=(8, 8))

    x = np.arange(-5, 5, 0.2)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=.01, size=len(x))
    obj.set_data_set(x, y)
    obj.plot_correlation_func(221)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=.1, size=len(x))
    obj.set_data_set(x, y)
    obj.plot_correlation_func(222)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=1, size=len(x))
    obj.set_data_set(x, y)
    obj.plot_correlation_func(223)
    y = 0.5 * x ** 2 + norm.rvs(1, scale=10, size=len(x))
    obj.set_data_set(x, y)
    obj.plot_correlation_func(224)
    filename = "corr_demo_2.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")

