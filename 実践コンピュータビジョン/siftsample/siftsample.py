# -*- coding: utf-8 -*-
import sys, os
sys.path.append('../scipysample')
sys.path.append('../matchsample')
# ここまでおまじない
import numpy, pylab
from scipy.misc import imsave
from scipysample import ScipySample
from matchsample import MatchSample

class SiftSample(ScipySample):
    def __init__(self, name = "sample_image.jpg"):
        ScipySample.__init__(self, name)
        self._image_name = name
        self._sift_name  = None
        self._sift_loc   = None
        self._sift_desc  = None

    def _process_image(self, resultname, params):
        """ 画像を処理してファイルに結果を保存する """
        if self._is_color:
            self.convert_grey()

        if self._image_name[-3:] != 'pgm': 
            # pgmファイルを作成する
            self._image_obj.save('tmp.pgm')
            self._image_name = 'tmp.pgm'

        cmmd = str("sift " + self._image_name + " --output=" + resultname + " " + params)
        os.system(cmmd)
        self._sift_name = resultname
        os.remove(self._image_name)

    def make_sift_feature(self, resultname = "image.sift", params = "--edge-thresh 10 --peak-thresh 5"):
        """ 特徴量を読み込んで行列形式で返す(データとしてすでにあるのでプロパティには定義しない) """
        if self._sift_name is None:
            self._process_image(resultname, params)
        f = numpy.loadtxt(self._sift_name)

        self._sift_loc  = f[:,:4] # 特徴点の配置と記述子
        self._sift_desc = f[:,4:]
        os.remove(self._sift_name)

    def get_shift_location(self):

        return self._sift_loc

    def get_sift_descriptors(self):

        return self._sift_desc

    def plot_sift_feature(self, name, circle = False):
        if self._sift_loc is None:
            self.make_sift_feature()
        """ 画像を特徴量とともに描画する。
        入力：im（配列形式の画像）、locs（各特徴量の座標とスケール、方向）"""

        def draw_circle(c,r):
            t = pylab.arange(0,1.01,.01) * 2 * numpy.pi
            x = r * numpy.cos(t) + c[0]
            y = r * numpy.sin(t) + c[1]
            pylab.plot(x, y, 'b', linewidth=2)

        pylab.figure(dpi=160)
        pylab.gray()
        pylab.imshow(self._array_image, aspect='auto')
        locs = self._sift_loc
        if circle:
            for p in locs:
                draw_circle(p[:2], p[2])
        else:
            pylab.plot(locs[:,0], locs[:,1], 'ob')
        pylab.axis('off')
        pylab.savefig(name, dpi=160)

class SiftMatch(MatchSample):
    def __init__(self, image_1 = SiftSample(), image_2 = SiftSample()):
        MatchSample.__init__(self, image_1, image_2)


    def match(self):
        """ 第1の画像の各記述子について、第2の画像の対応点を求める。
           入力：desc1（第1の画像の記述子）、desc2（第2の画像の記述子）"""
        
        if self._image_1.get_sift_descriptors() is None:
            self._image_1.make_sift_feature()
        desc1 = numpy.array([d / numpy.linalg.norm(d) for d in self._image_1.get_sift_descriptors()])
        if self._image_2.get_sift_descriptors() is None:
            self._image_2.make_sift_feature()
        desc2 = numpy.array([d / numpy.linalg.norm(d) for d in self._image_2.get_sift_descriptors()])

        dist_ratio  = 0.6
        desc1_size  = desc1.shape

        matchscores = numpy.zeros(desc1_size[0], 'int')
        desc2t      = desc2.T # あらかじめ転置行列を計算しておく

        for i in range(desc1_size[0]):
            dotprods = numpy.dot(desc1[i,:],desc2t) # 内積ベクトル
            dotprods = 0.9999 * dotprods
            # 第2の画像の特徴点の逆余弦を求め、ソートし、番号を返す
            indx = numpy.argsort(numpy.arccos(dotprods))

            # 最も近い近接点との角度が、2番目に近いもののdist_rasio倍以下か？
            if numpy.arccos(dotprods)[indx[0]] < dist_ratio * numpy.arccos(dotprods)[indx[1]]:
                matchscores[i] = int(indx[0])

        self._match_score = matchscores

    def plot_matches(self, name, show_below = True):
        """ 対応点を線で結んで画像を表示する
          入力： im1,im2（配列形式の画像）、locs1,locs2（特徴点座標）
             machescores（match()の出力）、
             show_below（対応の下に画像を表示するならTrue）"""
        im1 = self._image_1.get_array_image()
        im2 = self._image_2.get_array_image()
        self.appendimages()
        im3 = self._append_image
        if self._match_score is None:
            self.match()
        locs1 = self._image_1.get_shift_location()
        locs2 = self._image_2.get_shift_location()
        if show_below:
            im3 = numpy.vstack((im3,im3))
        pylab.figure(dpi=160)
        pylab.gray()
        pylab.imshow(im3, aspect = 'auto')

        cols1 = im1.shape[1]
        for i,m in enumerate(self._match_score):
            if m>0: pylab.plot([locs1[i][0],locs2[m][0]+cols1], [locs1[i][1],locs2[m][1]], 'c')
        pylab.axis('off')
        pylab.savefig(name, dpi=160)
    
