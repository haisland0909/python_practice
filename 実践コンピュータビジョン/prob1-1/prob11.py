# -*- coding: utf-8 -*-
import sys
sys.path.append('../pylabsample')
sys.path.append('../numpysample')
# ここまでおまじない
from numpysample import NumpySample
import pylabsample
from scipy.ndimage import filters

class Prob11(NumpySample):
    # ガウシアンフィルターを任意の標準偏差でかけた後輪郭を表示する(グレースケールのみ)
    def _add_gaussian_filter(self, sd):
        g_im = self._array_image
        if self._is_color:
            for i in range(3):
                g_im[:, :, i] = filters.gaussian_filter(self._array_image[:, :, i], sd)
        else:
            g_im = filters.gaussian_filter(self._array_image, sd)
        self._array_image = g_im
        self._convert_image()

    def create_gaussian_contour(self, sd, name):
        self.convert_grey()
        self._add_gaussian_filter(sd)
        pylabsample.create_contour(self._array_image, name)


