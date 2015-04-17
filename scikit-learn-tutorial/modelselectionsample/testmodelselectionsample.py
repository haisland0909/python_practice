import unittest, modelselectionsample, numpy
from sklearn import svm
import matplotlib.pyplot as plt

class Test_testmodelselectionsample(unittest.TestCase):
    def test_hold_out_validation(self):
        obj = modelselectionsample.CrossValidationSample()
        X   = [1,2,3,4]
        Y   = []
        Y.append(obj.hold_out_validation(50))
        Y.append(obj.hold_out_validation(100))
        Y.append(obj.hold_out_validation(200))
        Y.append(obj.hold_out_validation(400))
        plt.figure()
        plt.bar(X,Y, align="center")
        plt.title("hold_out_validation(total_data=%d)" % obj._x_digits.shape[0])
        plt.xlabel("test_data_num")
        plt.ylabel("fit_score")
        plt.xticks(X, ["50", "100", "200", "400"])
        plt.ylim(0, 1.1)
        plt.savefig("hold_out_validation.jpg")

    def test_k_fold_validation(self):
        obj = modelselectionsample.CrossValidationSample()
        print numpy.mean(obj.k_fold_validation())
        print numpy.mean(obj.k_fold_validation(4))
        print numpy.mean(obj.k_fold_validation(5))
        obj = modelselectionsample.CrossValidationSample()
        obj.set_fit_way(svm.SVC(C=1, kernel='poly'))
        print numpy.mean(obj.k_fold_validation())
        print numpy.mean(obj.k_fold_validation(4))
        print numpy.mean(obj.k_fold_validation(5))
        
if __name__ == '__main__':
    unittest.main()
