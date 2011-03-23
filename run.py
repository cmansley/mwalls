#!/usr/bin/env python

from blocking import BlockingWorld
from shortcut import ShortcutWorld
from alternating import AlternatingWorld
from dyna import DynaQ

def eval(outfile, env, epsilon, init):

    f = open(outfile, "w")

    output = []

    for j in range(50):
        
        cumrew = 0.0
        s = env.ss()
        alg = DynaQ(epsilon, init)
        
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
            f.write(str(output[j][i]) + ',')
        f.write('\n')

    f.close()


epsilons = [0, 0.001]
aname = ['m','p']
inits = [0, 0.9, 1, 2, 10]
envs = [BlockingWorld(), ShortcutWorld()]
ename = ['bl', 'sc']

for i in range(len(envs)):
    for j in range(len(epsilons)):
        for k in inits:
            filename = 'dyna_' + aname[j] + '_' + str(k) + '_' + ename[i] + '.txt'
            eval(filename, envs[i], epsilons[j], k)
