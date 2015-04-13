# -*- coding: utf-8 -*-
import sys, numpy
sys.path.append('../harrissample')
# ここまでおまじない
from harrissample import HarrisSample

class Prob22(HarrisSample):
    def make_harris_point_with_gaussian(self, sd):
        self._add_gaussian_filter(sd)
        self.make_harris_points()
