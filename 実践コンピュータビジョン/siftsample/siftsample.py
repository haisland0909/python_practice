# -*- coding: utf-8 -*-
import sys, os
sys.path.append('../scipysample')
sys.path.append('../matchsample')
# �����܂ł��܂��Ȃ�
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
        """ �摜���������ăt�@�C���Ɍ��ʂ�ۑ����� """
        if self._is_color:
            self.convert_grey()

        if self._image_name[-3:] != 'pgm': 
            # pgm�t�@�C�����쐬����
            self._image_obj.save('tmp.pgm')
            self._image_name = 'tmp.pgm'

        cmmd = str("sift " + self._image_name + " --output=" + resultname + " " + params)
        os.system(cmmd)
        self._sift_name = resultname
        os.remove(self._image_name)

    def make_sift_feature(self, resultname = "image.sift", params = "--edge-thresh 10 --peak-thresh 5"):
        """ �����ʂ�ǂݍ���ōs��`���ŕԂ�(�f�[�^�Ƃ��Ă��łɂ���̂Ńv���p�e�B�ɂ͒�`���Ȃ�) """
        if self._sift_name is None:
            self._process_image(resultname, params)
        f = numpy.loadtxt(self._sift_name)

        self._sift_loc  = f[:,:4] # �����_�̔z�u�ƋL�q�q
        self._sift_desc = f[:,4:]
        os.remove(self._sift_name)

    def get_shift_location(self):

        return self._sift_loc

    def get_sift_descriptors(self):

        return self._sift_desc

    def plot_sift_feature(self, name, circle = False):
        if self._sift_loc is None:
            self.make_sift_feature()
        """ �摜������ʂƂƂ��ɕ`�悷��B
        ���́Fim�i�z��`���̉摜�j�Alocs�i�e�����ʂ̍��W�ƃX�P�[���A�����j"""

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
        """ ��1�̉摜�̊e�L�q�q�ɂ��āA��2�̉摜�̑Ή��_�����߂�B
           ���́Fdesc1�i��1�̉摜�̋L�q�q�j�Adesc2�i��2�̉摜�̋L�q�q�j"""
        
        if self._image_1.get_sift_descriptors() is None:
            self._image_1.make_sift_feature()
        desc1 = numpy.array([d / numpy.linalg.norm(d) for d in self._image_1.get_sift_descriptors()])
        if self._image_2.get_sift_descriptors() is None:
            self._image_2.make_sift_feature()
        desc2 = numpy.array([d / numpy.linalg.norm(d) for d in self._image_2.get_sift_descriptors()])

        dist_ratio  = 0.6
        desc1_size  = desc1.shape

        matchscores = numpy.zeros(desc1_size[0], 'int')
        desc2t      = desc2.T # ���炩���ߓ]�u�s����v�Z���Ă���

        for i in range(desc1_size[0]):
            dotprods = numpy.dot(desc1[i,:],desc2t) # ���σx�N�g��
            dotprods = 0.9999 * dotprods
            # ��2�̉摜�̓����_�̋t�]�������߁A�\�[�g���A�ԍ���Ԃ�
            indx = numpy.argsort(numpy.arccos(dotprods))

            # �ł��߂��ߐړ_�Ƃ̊p�x���A2�Ԗڂɋ߂����̂�dist_rasio�{�ȉ����H
            if numpy.arccos(dotprods)[indx[0]] < dist_ratio * numpy.arccos(dotprods)[indx[1]]:
                matchscores[i] = int(indx[0])

        self._match_score = matchscores

    def plot_matches(self, name, show_below = True):
        """ �Ή��_����Ō���ŉ摜��\������
          ���́F im1,im2�i�z��`���̉摜�j�Alocs1,locs2�i�����_���W�j
             machescores�imatch()�̏o�́j�A
             show_below�i�Ή��̉��ɉ摜��\������Ȃ�True�j"""
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
    
