# -*- coding: utf-8 -*-
import sys
sys.path.append('../numpysample')
# ここまでおまじない
import numpy
from scipy.ndimage import filters
from scipy.misc import imsave
from numpysample import NumpySample

class ScipySample(NumpySample):
    def __init__(self, name = "sample_image.jpg"):
        NumpySample.__init__(self, name)
        self.reset_grad()

    def reset_grad(self):
        self._image_grad_x   = numpy.zeros(self._array_image.shape)
        self._image_grad_y   = numpy.zeros(self._array_image.shape)
        self._image_grad_mag = numpy.zeros(self._array_image.shape)

    def convert_grey(self):
        NumpySample.convert_grey(self)
        self.reset_grad()

    def calc_gradiation(self, sd):
        if self._is_color:
            for i in range(3):
                filters.gaussian_filter(self._array_image[:, :, i], (sd, sd), (0, 1), self._image_grad_x[:, :, i])
                filters.gaussian_filter(self._array_image[:, :, i], (sd, sd), (1, 0), self._image_grad_y[:, :, i])
                self._image_grad_mag[:, :, i] = numpy.sqrt(self._image_grad_x[:, :, i]**2 + self._image_grad_y[:, :, i]**2)
        else:
            filters.gaussian_filter(self._array_image, (sd, sd), (0, 1), self._image_grad_x)
            filters.gaussian_filter(self._array_image, (sd, sd), (1, 0), self._image_grad_y)
            self._image_grad_mag = numpy.sqrt(self._image_grad_x ** 2 + self._image_grad_y ** 2)

    def get_grad_x(self):

        return self._image_grad_x

    def get_grad_y(self):

        return self._image_grad_y

    def get_grad_mag(self):

        return self._image_grad_mag

    def save(self, name):
        imsave(name, self._array_image)
