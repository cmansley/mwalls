#!/usr/bin/env python

from blocking import BlockingWorld
from shortcut import ShortcutWorld
from dyna import DynaQ

# Experimental Framework 
env = BlockingWorld()
#env = ShortcutWorld()

output = []

for j in range(50):
    
    cumrew = 0.0
    s = env.ss()
    alg = DynaQ()

    output.append([])

    for i in range(env.len()):
        
        a = alg.policy(s)
    
        sp, r = env.step(s,a,i)

        alg.learn(s,a,r,sp)

        cumrew += r

        s = sp

        output[j].append(cumrew)


for i in range(env.len()):
    for j in range(50):
        print output[j][i],',',
    print
