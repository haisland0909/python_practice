import unittest, Chap3, numpy
from sklearn import svm
import matplotlib.pyplot as plt

class Test_testChap3(unittest.TestCase):
    def test_do_kmean(self):
        obj = Chap3.Chap3Sample()
        w   = 0.2
        Y_C = []
        Y_H = []
        Y_V = []
        X   = numpy.arange(3)
        for num in (10, 30, 50):
            obj.set_kmean_solver(num)
            obj.do_kmean()
            Y_H.append(obj.get_homo_score())
            Y_C.append(obj.get_comp_score())
            Y_V.append(obj.get_vmeasure_score())
        plt.figure()
        plt.bar(X,Y_H, align="center", label="homogeneity", width = w)
        plt.bar(X + w,Y_C, align="center", label="completeness", width = w, color='g')
        plt.bar(X + w * 2, Y_V, align="center", label="V-measure", width = w, color='y')
        plt.title("k-mean cluster score")
        plt.xlabel("cluster_num")
        plt.ylabel("score")
        plt.xticks(X + w, ["10", "30", "50"])
        plt.ylim(0, 0.6)
        plt.legend(loc="best")
        plt.savefig("k-mean.jpg")
        

if __name__ == '__main__':
    unittest.main()
