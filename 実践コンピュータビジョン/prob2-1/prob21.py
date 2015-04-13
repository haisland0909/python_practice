# -*- coding: utf-8 -*-
import sys, numpy
sys.path.append('../harrissample')
# ここまでおまじない
from harrissample import HarrisMatch

class Prob21(HarrisMatch):
    # 点の間の距離に最大値を付与する
    def harris_match(self ,threshold=0.5, max_rel = 0.9):
        """ 正規化相互相関を用いて、第1の画像の各コーナー点記述子について、
        第2の画像の対応点を選択する。"""
        if self._image_1.get_descriptors() is None:
            self._image_1.calc_descriptors()
        desc1 = self._image_1.get_descriptors()
        if self._image_2.get_descriptors() is None:
            self._image_2.calc_descriptors()
        desc2 = self._image_2.get_descriptors()
        n     = len(desc1[0])

        # 対応点ごとの距離
        d = -(numpy.ones((len(desc1), len(desc2))))
        for i in range(len(desc1)):
            for j in range(len(desc2)):
                d1 = (desc1[i] - numpy.mean(desc1[i])) / numpy.std(desc1[i])
                d2 = (desc2[j] - numpy.mean(desc2[j])) / numpy.std(desc2[j])
                ncc_value = numpy.sum(d1 * d2) / (n-1)
                if max_rel > ncc_value > threshold:
                    d[i,j] = ncc_value

        ndx               = numpy.argsort(-d)
        self._match_score = ndx[:,0]
