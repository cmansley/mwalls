from __future__ import division

import random

def learn(alpha):
    mu = 0.0
    for x in range(100):
        c = random.random()
        if c > 0.5:
            o = 1
        else:
            o = 0

        if alpha > 0:
            mu = (1-alpha)*mu + alpha*o
        else:
            a = 1/(x+1)
            mu = (1-a)*mu + a*o

        print o,mu


learn(0.5)
print "----"
learn(-1)
print "----"
learn(0.1)
