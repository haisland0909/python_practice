
# -*- coding: utf-8 -*-
import sys, numpy
sys.path.append('../scipysample')
sys.path.append('../pylabsample')
# ここまでおまじない
from scipysample import ScipySample
from pylabsample import create_hist
from PIL import Image
from scipy.ndimage import morphology, measurements

class Prob16(ScipySample):
    def __init__(self, name = "sample_image.jpg"):
        ScipySample.__init__(self, name)
        self._is_binary = False

    # 閾値処理によって画像をバイナリ化する
    def convert_binary(self, lim):
        if self._is_color:
            self.convert_grey()
        self._array_image = 255 * (self._array_image < lim)
        self._convert_image()
        self._is_binary   = True
    
    # ラベル割り当てます
    def get_label(self):
        self._array_image, num_obj = measurements.label(self._array_image)
        self._convert_image()

        return num_obj

    # オープニング演算
    def do_binary_opening(self, area = numpy.ones((3, 3)), iteration = 1):
        if self._is_binary:
            self._array_image = self._array_image / 255
            self._array_image = morphology.binary_opening(self._array_image, area, iterations = iteration)
            self._array_image = 255 * self._array_image
        else:
            self._array_image = morphology.grey_opening(self._array_image, area.shape)
        self._convert_image()
            