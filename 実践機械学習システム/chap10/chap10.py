# -*- coding: utf-8 -*-
# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License
import numpy as np
import mahotas as mh
from sklearn import cross_validation
from sklearn.linear_model.logistic import LogisticRegression
from glob import glob
from features import texture, edginess_sobel

class ImageBase(object):
    def __init__(self, name):
        self._image = mh.imread(name)

class ImageThreshold(ImageBase):
    def __init__(self, name):
        super(ImageThreshold, self).__init__(name)
        self._gray_image     = mh.colors.rgb2gray(self._image, dtype=np.uint8)
        self._otsu_thresh    = 0
        self._rc_thresh      = 0
        self._gaussian_image = None
        mh.imsave("threshold-base.jpeg", self._image)

    def otsu_threshold(self):
        self._otsu_thresh = mh.thresholding.otsu(self._gray_image)

    def set_gray_image(self, image):
        self._gray_image = image.astype(np.uint8)

    def get_otsu_thresh(self, calc = False):
        if self._otsu_thresh == 0 or calc:
            self.otsu_threshold()

        return self._otsu_thresh

    def rc_threshold(self):
        self._rc_thresh = mh.thresholding.rc(self._gray_image)

    def get_rc_thresh(self):
        if self._rc_thresh == 0:
            self.rc_threshold()

        return self._rc_thresh

    def get_binary_image(self, threshold):

        return (self._gray_image > threshold)

    def save_image(self, name, image):
        mh.imsave(name, image.astype(np.uint8))

    def save_open_image(self, threshold, name):
        image = mh.open(self.get_binary_image(threshold), np.ones((15, 15)))
        mh.imsave(name, image.astype(np.uint8) * 255)

    def save_binary_image(self, threshold, name):
        mh.imsave(name, self.get_binary_image(threshold).astype(np.uint8) * 255)

    def get_gausian_filtered_image(self, std = None):
        if std is not None:
            self._gaussian_image = mh.gaussian_filter(self._gray_image,  std)

        return self._gaussian_image

class ImageFilter(ImageBase):
    def __init__(self):
        self._image          = mh.demos.load('lena')
        self._gray_image     = mh.colors.rgb2gray(self._image, dtype=np.uint8)
        self._gaussian_image = None
        mh.imsave("filter-base.jpeg", self._image)

    def get_salt_pepper_image(self):
        im     = self._gray_image
        salt   = np.random.random(im.shape) > .975
        pepper = np.random.random(im.shape) > .975
        
        # salt is 170 & pepper is 30
        # Some playing around showed that setting these to more extreme values looks
        # very artificial. These look nicer
        
        im = np.maximum(salt * 170, mh.stretch(im))
        im = np.minimum(pepper * 30 + im * (~pepper), im)

        return im

    def create_ring(self):
        im = self._image
        # This breaks up the image into RGB channels
        r, g, b = im.transpose(2, 0, 1)
        h, w = r.shape
        
        # smooth the image per channel:
        r12 = mh.gaussian_filter(r, 12.)
        g12 = mh.gaussian_filter(g, 12.)
        b12 = mh.gaussian_filter(b, 12.)
        
        # build back the RGB image
        im12 = mh.as_rgb(r12, g12, b12)
        
        X, Y = np.mgrid[:h, :w]
        X = X - h / 2.
        Y = Y - w / 2.
        X /= X.max()
        Y /= Y.max()
        
        # Array C will have the highest values in the center, fading out to the edges:
        
        C = np.exp(-2. * (X ** 2 + Y ** 2))
        C -= C.min()
        C /= C.ptp()
        C = C[:, :, None]
        
        # The final result is sharp in the centre and smooths out to the borders:
        ring = mh.stretch(im * C + (1 - C) * im12)

        return ring

    def save_image(self, name, image):
        mh.imsave(name, image.astype(np.uint8))


if __name__ == "__main__":
    # Filter関係の処理
    '''
    obj = ImageFilter()
    im  = obj.get_salt_pepper_image()
    obj.save_image("salt_pepper.jpg", im)
    ring = obj.create_ring()
    obj.save_image("ring.jpg", ring)
    '''
    # 閾値関係の処理   
    '''
    obj  = ImageThreshold('./SimpleImageDataset/building05.jpg')
    otsu = obj.get_otsu_thresh()
    obj.save_binary_image(otsu, 'otsu-threshold.jpeg')
    obj.save_open_image(otsu, 'otsu-closed.jpeg')
    rc   = obj.get_rc_thresh()
    obj.save_binary_image(rc, 'rc-threshold.jpeg')
    g_8  = obj.get_gausian_filtered_image(8)
    obj.save_image("gaussian_8.jpeg", g_8)
    g_16  = obj.get_gausian_filtered_image(16)
    obj.save_image("gaussian_16.jpeg", g_16)
    g_24  = obj.get_gausian_filtered_image(24)
    obj.save_image("gaussian_24.jpeg", g_24)
    obj.set_gray_image(g_8)
    otsu_2 = obj.get_otsu_thresh(True)
    obj.save_binary_image(otsu_2, 'otsu-gaussian-threshold.jpeg')
    '''


