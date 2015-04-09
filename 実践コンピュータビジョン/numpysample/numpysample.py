# -*- coding: utf-8 -*-
import sys
sys.path.append('../pilsample')
# ここまでおまじない

import numpy
from pilsample import PilSample
from PIL import Image

class NumpySample(PilSample):
    def __init__(self, name = "sample_image.jpg"):
        PilSample.__init__(self, name)
        self._array_image = numpy.array(self._image_obj)

    def _convert_image(self):
        self._image_obj = Image.fromarray(numpy.uint8(self._array_image))

    def _convert_array(self):
        self._array_image = numpy.array(self._image_obj)

    def convert_grey(self):
        PilSample.convert_grey(self)
        self._convert_array()
    # 画像を行列形式のデータとして返却
    def get_array_image(self):

        return self._array_image
    # グレーレベルの変換
    def change_color_level(self, f):
        im = self._array_image
        if self._is_color:
            new_im = im
            for i in range(3):
                new_im[:, :, i] = f(im[:, :, i])
        else:
            new_im = f(im)
        self._array_image = new_im
        self._convert_image()
    # リサイズ
    def resize_image(self, sz):
        im               = self._image_obj
        new_im           = im.resize(sz)
        self._image_obj  = new_im
        self._convert_array() 


