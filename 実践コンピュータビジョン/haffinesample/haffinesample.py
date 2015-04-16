# -*- coding: utf-8 -*-
import sys
sys.path.append('../harrissample')
# ここまでおまじない
import numpy, pylab
from scipy.ndimage import filters
from harrissample import HarrisSample

class HaffinSample(HarrisSample):
    def __init__(self, name = "sample_image.jpg"):
        HarrisSample.__init__(self, name)
