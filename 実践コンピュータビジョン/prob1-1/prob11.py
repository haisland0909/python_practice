# -*- coding: utf-8 -*-
import sys
sys.path.append('../pylabsample')
sys.path.append('../scipysample')
# ここまでおまじない
from scipysample import ScipySample
import pylabsample
from scipy.ndimage import filters

class Prob11(ScipySample):
    # ガウシアンフィルターを任意の標準偏差でかけた後輪郭を表示する(グレースケールのみ)
    def create_gaussian_contour(self, sd, name):
        self.convert_grey()
        self._add_gaussian_filter(sd)
        pylabsample.create_contour(self._array_image, name)


