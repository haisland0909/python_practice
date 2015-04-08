# -*- coding: utf-8 -*-
import sys
sys.path.append('../pylabsample')
sys.path.append('../numpysample')
# ここまでおまじない
import pylabsample, numpysample
from scipy.ndimage import filters
# ガウシアンフィルターを任意の標準偏差でかけた後輪郭を表示する
def create_gaussian_contour(sd, name):
    im   = numpysample.get_sample_image_array("L")
    g_im = filters.gaussian_filter(im, sd)
    pylabsample.create_contour(g_im, name)

    return g_im

