# coding:cp932
# 比較する関数を渡してソート
def qsort(filter, lst):
    if lst == []:

        return lst
    else:
        def partition(x, lst, filter=True): # x を基準に lst を「True」と「False」に分割する
            a, b = [], []
            # フィルターに通してT or Fでふるいにかける
            for i in lst:
                if filter(i, x):       
                    a.append(i) 
                else:
                    b.append(i) 

            return a, b

    xs, ys = partition(lst[0], lst[1:], filter)

    # 再帰でリストが空になるまで
    return qsort(filter, xs) + [lst[0]] + qsort(filter, ys)

# 文字列の長さでBoolean
slencmp = lambda x, y: len(x) < len(y)
