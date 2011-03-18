#!/usr/bin/env python

from blocking import BlockingWorld
from dyna import DynaQ

# Experimental Framework 
env = BlockingWorld()
alg = DynaQ()

output = []

for j in range(50):
    
    cumrew = 0.0
    s = env.ss()

    output.append([])

    for i in range(3000):
        
        a = alg.policy(s)
    
        sp, r = env.step(s,a,i)

        alg.learn(s,a,r,sp)

        cumrew += r

        s = sp

        output[j].append(cumrew)


for i in range(3000):
    for j in range(50):
        print output[j][i],',',
    print
