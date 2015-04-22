# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
from load_ml100k import load
from abc import ABCMeta, abstractmethod
from sklearn.linear_model import ElasticNetCV
from sklearn.cross_validation import KFold


class AbstractEstimateBase(object):
    __metaclass__ = ABCMeta
    reviews       = None
    def __init__(self):
        if AbstractEstimateBase.reviews is None:
            AbstractEstimateBase.reviews = load()

    @abstractmethod
    def all_estimates(self):
        print "abstract"

class UserModelEstimate(AbstractEstimateBase):
    def learn_for(self, i):
        reviews        = AbstractEstimateBase.reviews
        reg            = ElasticNetCV(fit_intercept=True, alphas=[
                           0.0125, 0.025, 0.05, .125, .25, .5, 1., 2., 4.])
        nusers,nmovies = reviews.shape
        u              = reviews[i]
        us             = np.arange(reviews.shape[0])
        us             = np.delete(us, i)
        ps,            = np.where(u.ravel() > 0)
        x              = reviews[us][:, ps].T
        kf             = KFold(len(ps), n_folds=4)
        predictions    = np.zeros(len(ps))
        for train, test in kf:
            xc = x[train].copy()
            x1 = np.array([xi[xi > 0].mean() for xi in xc])
            x1 = np.nan_to_num(x1)

            for i in range(xc.shape[0]):
                xc[i] -= (xc[i] > 0) * x1[i]

            reg.fit(xc, u[train] - x1)

            xc = x[test].copy()
            x1 = np.array([xi[xi > 0].mean() for xi in xc])
            x1 = np.nan_to_num(x1)

            for i in range(xc.shape[0]):
                xc[i] -= (xc[i] > 0) * x1[i]

            p = reg.predict(xc).ravel()
            predictions[test] = p
        fill_preds = np.zeros(nmovies)
        fill_preds[ps] = predictions

        return fill_preds

    def all_estimates(self):
        reviews    = AbstractEstimateBase.reviews
        whole_data = []
        for i in range(reviews.shape[0]):
            s = self.learn_for(i)
            whole_data.append(s)

        return np.array(whole_data)

class SimilarMovieEstimate(AbstractEstimateBase):
    def nn_movie(self, ureviews, reviews, uid, mid, k=1):
        '''Movie neighbor based classifier
        Parameters
        ----------
        ureviews : ndarray
        reviews : ndarray
        uid : int
            index of user
        mid : int
            index of movie
        k : int
            index of neighbor to return    
        Returns
        -------
        pred : float
        '''
        X = ureviews
        y = ureviews[mid].copy()
        y -= y.mean()
        y /= (y.std() + 1e-5)
        corrs = np.dot(X, y)
        likes = corrs.argsort()
        likes = likes[::-1]
        c = 0
        pred = 3.
        for ell in likes:
            if ell == mid:
                continue
            if reviews[uid, ell] > 0:
                pred = reviews[uid, ell]
                if c == k:
                    return pred
                c += 1
        return pred
    
    
    def all_estimates(self, k=1):
        '''Estimate all review ratings
        '''
        reviews = AbstractEstimateBase.reviews.astype(float)
        k -= 1
        nusers, nmovies = reviews.shape
        estimates = np.zeros_like(reviews)
        for u in range(nusers):
            ureviews = np.delete(reviews, u, axis=0)
            ureviews -= ureviews.mean(0)
            ureviews /= (ureviews.std(0) + 1e-5)
            ureviews = ureviews.T.copy()
            for m in np.where(reviews[u] > 0)[0]:
                estimates[u, m] = self.nn_movie(ureviews, reviews, u, m, k)

        return estimates

class Chap8(object):
    def __init__(self):
        self._usermodel    = UserModelEstimate()
        self._similarmovie = SimilarMovieEstimate()

    def do_stack_learn(self):
        reviews = AbstractEstimateBase.reviews
        # Collect several estimates
        es = np.array([
            self._usermodel.all_estimates(),
            self._similarmovie.all_estimates(k = 1),
            self._similarmovie.all_estimates(k = 2),
            self._similarmovie.all_estimates(k = 3),
            self._similarmovie.all_estimates(k = 4),
            self._similarmovie.all_estimates(k = 5),
        ])
        
        total_error = 0.0
        coefficients = []
        
        reg = LinearRegression()
        # Iterate over all users
        for u in range(reviews.shape[0]):
            es0 = np.delete(es, u, axis=1)
            r0 = np.delete(reviews, u, axis=0)
            X, Y = np.where(r0 > 0)
            X = es[:, X, Y]
            y = r0[r0 > 0]
            reg.fit(X.T, y)
            coefficients.append(reg.coef_)
        
            r0 = reviews[u]
            X = np.where(r0 > 0)
            p0 = reg.predict(es[:, u, X].squeeze().T)
            err0 = r0[r0 > 0] - p0
            total_error += np.dot(err0, err0)
        coefficients = np.array(coefficients)

        print coefficients

if __name__ == '__main__':
    obj = Chap8()
    obj.do_stack_learn()
