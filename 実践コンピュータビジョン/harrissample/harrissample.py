# -*- coding: utf-8 -*-
import sys
sys.path.append('../scipysample')
# ここまでおまじない
import numpy, pylab
from scipy.ndimage import filters
from scipy.misc import imsave
from scipysample import ScipySample

class HarrisSample(ScipySample):
    def __init__(self, name = "sample_image.jpg"):
        ScipySample.__init__(self, name)
        self._harris_response     = None
        self._harris_points       = None
        self._harriss_descriptors = None


    def compute_harris_response(self, sigma=3):
        if self._is_color:
            self.convert_grey()
        """ グレースケール画像の各ピクセルについて
        Harrisコーナー検出器の応答関数を定義する """
        self.calc_gradiation(sigma)
        # 微分係数
        imx = self._image_grad_x
        imy = self._image_grad_y

        # Harris行列の成分を計算する
        Wxx = filters.gaussian_filter(imx*imx, sigma)
        Wxy = filters.gaussian_filter(imx*imy, sigma)
        Wyy = filters.gaussian_filter(imy*imy, sigma)

        # 判別式と対角成分
        Wdet = Wxx*Wyy - Wxy**2
        Wtr  = Wxx + Wyy
        numpy.seterr(divide='ignore', invalid='ignore')
        self._harris_response = numpy.nan_to_num(Wdet / Wtr)

    def make_harris_points(self, min_dist=10, threshold=0.1):
        """ Harris応答画像からコーナーを返す。
           min_distはコーナーや画像境界から分離する最小ピクセル数 """

        if self._harris_response is None:
            self.compute_harris_response()
        
        # 閾値thresholdを超えるコーナー候補を見つける
        harrisim         = self._harris_response
        corner_threshold = harrisim.max() * threshold
        harrisim_t       = (harrisim > corner_threshold) * 1
        
        # 候補の座標を得る
        coords = numpy.array(harrisim_t.nonzero()).T
        
        # 候補の値を得る
        candidate_values = [harrisim[c[0],c[1]] for c in coords]
        
        # 候補をソートする(候補として良い点からから順に検討)
        index = numpy.argsort(candidate_values)
        
        # 許容する点の座標を配列に格納する
        allowed_locations = numpy.zeros(harrisim.shape)
        allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
        
        # 最小距離を考慮しながら、最良の点を得る
        filtered_coords = []
        for i in index:
            if allowed_locations[coords[i,0],coords[i,1]] == 1:
                filtered_coords.append(coords[i])
                allowed_locations[(coords[i,0] - min_dist):(coords[i,0] + min_dist),
                    (coords[i,1] - min_dist):(coords[i,1] + min_dist)] = 0
        self._harris_points = filtered_coords

    def get_harris_response(self):

        return self._harris_response

    def get_harris_point(self):

        return self._harris_points

    def plot_harris_points(self, name):
        """ 画像に見つかったコーナーを描画 """
        if self._harris_points is None:
            self.make_harris_points()
        pylab.figure(dpi=160)
        pylab.gray()
        pylab.imshow(self._array_image, aspect='auto')
        pylab.plot([p[1] for p in self._harris_points],[p[0] for p in self._harris_points],'*')
        pylab.axis('off')
        pylab.savefig(name, dpi=160)

    def calc_descriptors(self, wid=5):
        """ 各Harriコーナ点について、点の周辺で幅 2*wid+1 の近傍ピクセル値を返す。
            （点の最小距離 min_distance > wid を仮定する）"""
        if self._harris_points is None:
            self.make_harris_points()
        image = self._array_image
        desc = []
        for coords in self._harris_points:
            patch = image[coords[0]-wid:coords[0]+wid+1,
                        coords[1]-wid:coords[1]+wid+1].flatten() 
            desc.append(patch)
        
        self._harriss_descriptors = desc

    def get_descriptors(self):
        
        return self._harriss_descriptors

class HarrisMatch(object):

    def __init__(self, image_1 = HarrisSample(), image_2 = HarrisSample()):
        self._image_1      = image_1
        self._image_2      = image_2
        self._append_image = None
        self._match_score  = None

    def harris_match(self ,threshold=0.5):
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
                if ncc_value > threshold:
                    d[i,j] = ncc_value

        ndx               = numpy.argsort(-d)
        self._match_score = ndx[:,0]

    def get_harris_match_score(self):

        return self._match_score

    def appendimages(self):
        """ 2つの画像を左右に並べた画像を返す """
        im1 = self._image_1.get_array_image()
        im2 = self._image_2.get_array_image()

        # 行の少ない方を選び、空行を0で埋める
        rows1 = im1.shape[0]
        rows2 = im2.shape[0]
        
        if rows1 < rows2:
            im1 = numpy.concatenate((im1, numpy.zeros((rows2-rows1,im1.shape[1]))), axis=0)
        elif rows1 > rows2:
            im2 = numpy.concatenate((im2, numpy.zeros((rows1-rows2,im2.shape[1]))), axis=0)
        # 行が同じなら、0で埋める必要はない    
        self._append_image = numpy.concatenate((im1,im2), axis=1)

    def get_append_image(self):
        
        return self._append_image

    def save_append_image(self, name):
        imsave(name, self._append_image)

    def plot_matches(self, name = "harris_match.jpg", show_below = True, match_maximum = None):
        """ 対応点を線で結んで画像を表示する
        入力： 
        show_below（対応の下に画像を表示するならTrue）"""
        if self._append_image is None:
            self.appendimages()
        im1   = self._image_1.get_array_image()
        im2   = self._image_2.get_array_image()
        im3   = self._append_image
        if self._image_1.get_harris_point() is None:
            self._image_1.make_harris_points()
        if self._image_2.get_harris_point() is None:
            self._image_2.make_harris_points()
        locs1 = self._image_1.get_harris_point()
        locs2 = self._image_2.get_harris_point()
        if show_below:
            im3 = numpy.vstack((im3,im3))
        pylab.figure(dpi=160)
        pylab.gray()
        pylab.imshow(im3)
        
        cols1 = im1.shape[1]
        if self._match_score is None:
            self.harris_match()
        if match_maximum is not None:
            self._match_score = self._match_score[:match_maximum]
        for i,m in enumerate(self._match_score):
            if m>0: 
                pylab.plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
        pylab.axis('off')
        pylab.savefig(name, dpi=160)