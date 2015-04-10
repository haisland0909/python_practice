# -*- coding: utf-8 -*-
import sys, numpy
sys.path.append('../scipysample')
# ここまでおまじない
from scipysample import ScipySample
from PIL import Image

class Prob14(ScipySample):
    # 画像の勾配より輪郭を抽出する
    def create_edge(self):
        self.calc_gradiation(1)
        mag = self.get_grad_mag()

        return Image.fromarray(255 - numpy.uint8(self.get_grad_mag()))



