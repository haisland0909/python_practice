print('Hello World')
# -*- coding: utf-8 -*-
import sys
sys.path.append('../prob1-1')
# ここまでおまじない
from prob11 import Prob11

class Prob12(Prob11):
    # アンシャープマスク処理をかける
    def add_unsharped_mask(self, sd):
        old_im            = self._array_image.copy()
        self._add_gaussian_filter(sd)
        new_im            = old_im + (old_im - self._array_image)
        self._array_image = new_im
        self._convert_image()
