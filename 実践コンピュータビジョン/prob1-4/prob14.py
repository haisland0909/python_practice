# -*- coding: utf-8 -*-
import sys, numpy
sys.path.append('../scipysample')
# �����܂ł��܂��Ȃ�
from scipysample import ScipySample
from PIL import Image

class Prob14(ScipySample):
    # �摜�̌��z���֊s�𒊏o����
    def create_edge(self):
        self.calc_gradiation(1)
        mag = self.get_grad_mag()

        return Image.fromarray(255 - numpy.uint8(self.get_grad_mag()))



