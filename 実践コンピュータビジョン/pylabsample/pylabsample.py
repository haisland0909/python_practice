# -*- coding: utf-8 -*-
# ここまでおまじない
import pylab

# 等高線の表示
def create_contour(im, name):
    pylab.figure()
    pylab.contour(im, origin='image')
    pylab.axis('equal')
    pylab.axis('off')
    pylab.savefig(name)

