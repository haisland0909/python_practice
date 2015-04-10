# -*- coding: utf-8 -*-
import sys, numpy
sys.path.append('../scipysample')
# ‚±‚±‚Ü‚Å‚¨‚Ü‚¶‚È‚¢
from scipysample import ScipySample
from PIL import Image

class Prob14(ScipySample):
    # ‰æ‘œ‚ÌŒù”z‚æ‚è—ÖŠs‚ğ’Šo‚·‚é
    def create_edge(self):
        self.calc_gradiation(1)
        mag = self.get_grad_mag()

        return Image.fromarray(255 - numpy.uint8(self.get_grad_mag()))



