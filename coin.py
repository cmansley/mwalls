from __future__ import division

import collections
import random


t = collections.defaultdict(int)

for i in range(1000):
    r = random.random()

    if r < 0.1:
        o = 1
    elif r < 0.2:
        o = 2
    else:
        o = 3

    print o

    #alpha = 1/(i+1)
    alpha = 0.9

    for key in t.keys():
        t[key] = (1-alpha)*t[key]

    t[o] = t[o] + alpha


print 'Pre-normal:'
print t

n = sum(t.values())

for i in t.keys():
    t[key] = t[key]/n

print 'Post-normal:',n
print t

