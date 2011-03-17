#!/usr/bin/env python

from blocking import BlockingWorld
from dyna import DynaQ

# Experimental Framework 
env = BlockingWorld()
alg = DynaQ()

cumrew = 0.0
s = env.ss()

for i in range(3000):
        
    a = alg.policy(s)
    
    sp, r = env.step(s,a,i)

    alg.learn(s,a,r,sp)

    cumrew += r

    s = sp

    print cumrew
