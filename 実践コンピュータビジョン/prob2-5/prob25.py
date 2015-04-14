# -*- coding: utf-8 -*-
import sys, os
sys.path.append('../siftsample')
# ここまでおまじない
import numpy, pylab
from siftsample import SiftSample

class Prob25(SiftSample):
    def _process_image(self, resultname, params):
        """ 画像を処理してファイルに結果を保存する """
        if self._is_color:
            self.convert_grey()

        if self._image_name[-3:] != 'pgm': 
            # pgmファイルを作成する
            self._image_obj.save('tmp.pgm')
            self._image_name = 'tmp.pgm'
        m_cmmd = str("mser " + self._image_name + " --frames=tmp.mser")
        s_cmmd = str("sift " + self._image_name +" --output=" + resultname + " " + params + "read-frames=tmp.mser")
        os.system(m_cmmd)
        os.system(s_cmmd)
        self._sift_name = resultname
        os.remove(self._image_name)
        os.remove("tmp.mser")
