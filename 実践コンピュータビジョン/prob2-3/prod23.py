# -*- coding: utf-8 -*-
import sys, numpy, pylab
sys.path.append('../scipysample')
# ここまでおまじない
from scipysample import ScipySample
import fast9

class Prob23(ScipySample):
    def __init__(self, name = "sample_image.jpg"):
        ScipySample.__init__(self, name)
        self._fast_points       = None

    def make_fast_points(self, threshold = 20, min_dist = 10):
        if self._is_color:
            self.convert_grey()
        # detectで得られる座標は反対かもしれない
        corners, scores = fast9.detect(self._array_image, threshold)
        index           = numpy.argsort(scores)
        allowed_locations = numpy.zeros(self._array_image.shape)
        allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
        # 最小距離を考慮しながら、最良の点を得る
        filtered_coords = []
        for i in index:
            if allowed_locations[corners[i][1],corners[i][0]] == 1:
                filtered_coords.append((corners[i][1], corners[i][0]))
                allowed_locations[(corners[i][1] - min_dist):(corners[i][1] + min_dist),
                    (corners[i][0] - min_dist):(corners[i][0] + min_dist)] = 0
        self._fast_points = filtered_coords

    def get_fast_points(self):

        return self._fast_points


    def plot_fast_points(self, name):
        """ 画像に見つかったコーナーを描画 """
        if self._fast_points is None:
            self.make_fast_points()
        pylab.figure(dpi=160)
        pylab.gray()
        pylab.imshow(self._array_image, aspect='auto')
        pylab.plot([p[1] for p in self._fast_points],[p[0] for p in self._fast_points],'*')
        pylab.axis('off')
        pylab.savefig(name, dpi=160)