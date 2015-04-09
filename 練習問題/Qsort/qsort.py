# coding:cp932
# ��r����֐���n���ă\�[�g
def qsort(filter, lst):
    if lst == []:

        return lst
    else:
        def partition(x, lst, filter=True): # x ����� lst ���uTrue�v�ƁuFalse�v�ɕ�������
            a, b = [], []
            # �t�B���^�[�ɒʂ���T or F�łӂ邢�ɂ�����
            for i in lst:
                if filter(i, x):       
                    a.append(i) 
                else:
                    b.append(i) 

            return a, b

    xs, ys = partition(lst[0], lst[1:], filter)

    # �ċA�Ń��X�g����ɂȂ�܂�
    return qsort(filter, xs) + [lst[0]] + qsort(filter, ys)

# ������̒�����Boolean
slencmp = lambda x, y: len(x) < len(y)
