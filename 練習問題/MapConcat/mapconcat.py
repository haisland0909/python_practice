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

print mapconcat(str, ["foo", "bar", "baz"], "-")
print mapconcat(str, [1, 2, 3], " ")
print mapconcat(lambda c: c*3, "abc", "")
print mapconcat(lambda s: s.rjust(10), ["foo", "bar", "baz"], "")

