# coding:utf-8
from random import randint
# —”ƒŠƒXƒg‚ğì¬
def randlist(x = 1, upper = 100, lower = 0):
    a = []
    for var in range(0, x):
        insertNum = randint(lower, upper)
        a.append(insertNum)
    print a

randlist(10)
randlist(3, upper=50)  
randlist(6, lower=20, upper=50)  
