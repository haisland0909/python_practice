from random import randint

def randlist(x = 1):
    a = []
    for var in range(0, x):
        insertNum = randint(1, 100)
        a.append(insertNum)
    print a

randlist(10)
randlist(5)
randlist(3)