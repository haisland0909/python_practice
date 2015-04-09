# -*- coding: utf-8 -*-
import sys, numpy
sys.path.append('../prob1-1')
# ここまでおまじない
from prob11 import Prob11

class Prob13(Prob11):
    # Self-Quotient画像を作成する
    def make_self_quotient(self, sd):
        old_im            = self._array_image.copy()
        self._add_gaussian_filter(sd)
        new_im            = self._array_image.copy()
        if self._is_color:
            for i in range(3):
                new_im[:, :, i] = old_im[:, :, i] / new_im[:, :, i]
                new_im[:, :, i] = 255.0 * new_im[:, :, i] / numpy.amax(new_im[:, :, i])
        else:
            new_im = old_im / new_im
            new_im = 255.0 * new_im / numpy.amax(new_im)
        self._array_image = new_im
        self._convert_image()
