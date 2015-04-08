# coding:cp932
from random import randint
# 乱数リストを作成
def randlist(x = 1, upper = 100, lower = 0):
    a = []
    for var in range(0, x):
        insertNum = randint(lower, upper)
        a.append(insertNum)
    return a

