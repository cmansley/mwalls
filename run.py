#!/usr/bin/env python

import progressbar as pb

from blocking import BlockingWorld
from shortcut import ShortcutWorld
from alternating import AlternatingWorld
from dyna import DynaQ

def eval(outfile, env, epsilon, k):

    f = open(outfile, "w")

    output = []

    for j in range(50):
        
        cumrew = 0.0
        s = env.ss()
        alg = DynaQ(epsilon, k)
        
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
k = [10, 20, 50, 100]
#envs = [BlockingWorld(), ShortcutWorld()]
#ename = ['bl', 'sc']
envs = [AlternatingWorld()]
ename = ['al']

maxlen = len(envs)*len(epsilons)*len(k)
widgets = [pb.Bar('>'), ' ', pb.ETA(), ' ', pb.ReverseBar('<')]
pbar = pb.ProgressBar(widgets=widgets, maxval=maxlen).start()

count = 0
for x in range(len(envs)):
    for y in range(len(epsilons)):
        for z in k:
            filename = 'dyna_' + aname[y] + '_' + str(z) + '_' + ename[x] + '.txt'
            eval(filename, envs[x], epsilons[y], z)

            count += 1
            pbar.update(count)

pbar.finish()
