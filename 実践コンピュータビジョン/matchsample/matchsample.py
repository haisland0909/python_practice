# -*- coding: utf-8 -*-
import sys
sys.path.append('../scipysample')
# ここまでおまじない
import numpy, pylab
from scipy.ndimage import filters
from scipy.misc import imsave
from scipysample import ScipySample

class MatchSample(object):

    def __init__(self, image_1 = ScipySample(), image_2 = ScipySample()):
        self._image_1      = image_1
        self._image_2      = image_2
        self._append_image = None
        self._match_score  = None

    def match(self ,threshold=0.5):
        raise NotImplementedError

    def get_match_score(self):

        return self._match_score

    def appendimages(self):
        """ 2つの画像を左右に並べた画像を返す """
        im1 = self._image_1.get_array_image()
        im2 = self._image_2.get_array_image()

        # 行の少ない方を選び、空行を0で埋める
        rows1 = im1.shape[0]
        rows2 = im2.shape[0]
        
        if rows1 < rows2:
            im1 = numpy.concatenate((im1, numpy.zeros((rows2-rows1,im1.shape[1]))), axis=0)
        elif rows1 > rows2:
            im2 = numpy.concatenate((im2, numpy.zeros((rows1-rows2,im2.shape[1]))), axis=0)
        # 行が同じなら、0で埋める必要はない    
        self._append_image = numpy.concatenate((im1,im2), axis=1)

    def get_append_image(self):
        
        return self._append_image

    def save_append_image(self, name):
        imsave(name, self._append_image)

    def plot_matches(self):
        raise NotImplementedError
