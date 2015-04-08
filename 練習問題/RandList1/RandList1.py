# coding:cp932
from random import randint
# 乱数リストを作成
def randlist(x = 1):
    a = []
    for var in range(0, x):
        insertNum = randint(1, 100)
        a.append(insertNum)
    return a