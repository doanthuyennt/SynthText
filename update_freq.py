#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
import pickle
import collections
cnt = 0
filename = './new_dict.txt'
with open(filename) as f:
    c = Counter()
    for x in f:
        # x = x.decode('utf-8')
        c += Counter(x.strip())
        cnt += len(x.strip())
        # print c
print(cnt)

c = {k: v for k, v in sorted(c.items(), key=lambda item: item[0], reverse=True)}
# od = collections.OrderedDict(sorted(c.keys()))

for key in c:
    c[key] = float(c[key]) / cnt
    print( key, round(c[key]*100,7)," %")

d = dict(c)
# print d
with open("char_freq.cp", 'wb') as f:
    pickle.dump(d, f)