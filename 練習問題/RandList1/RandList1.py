# coding:cp932
from random import randint
# �������X�g���쐬
def randlist(x = 1):
    a = []
    for var in range(0, x):
        insertNum = randint(1, 100)
        a.append(insertNum)
    return a