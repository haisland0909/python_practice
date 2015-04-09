# -*- coding: utf-8 -*-
from __future__ import print_function
from PIL import Image

class PilSample(object):
    NAME_SAMPLE_IMAGE = "sample_image.jpg"

    def __init__(self, name = "sample_image.jpg"):
        self._image_obj = Image.open("../CommonImage/%s" % name)
        self._is_color  = True


    def get_image(self):

        return self._image_obj

    def convert_grey(self):
        if self._is_color:
            im              = self._image_obj
            im              = im.convert("L")
            self._image_obj = im
            self._is_color  = False

    def is_color(self):
        
        return self._is_color

    def save(self, name):
        self._image_obj.save(name)




