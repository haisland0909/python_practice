def mapconcat(function, sequence, separator):
    res  = ""
    leng = len(sequence)
    for i in range(0, leng):
        s   = sequence[i]
        s   = function(s)
        res = res + s
        if i < leng - 1:
            res = res + separator

    return res

