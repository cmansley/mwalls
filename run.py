#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import progressbar as pb

from blocking import BlockingWorld
from shortcut import ShortcutWorld
from alternating import AlternatingWorld
from dyna import DynaQ
from powerdyna import PowerDynaQ

num_avg = 10

def eval(outfile, env, epsilon, k, alpha):

    f = open(outfile, "w")

    output = []

    for j in range(num_avg):

        # reset environment/scores
        cumrew = 0.0
        s = env.ss()
        alg = PowerDynaQ(k, alpha)
        #alg = DynaQ(epsilon, k)
        
        output.append([])

        # execute environment
        for i in range(env.len()):
            # grab action
            a = alg.policy(s)

            # grab next state
            sp, r = env.step(s,a,i)

            # pass next state to learner
            alg.learn(s,a,r,sp)

            if i < 4000 and False:
                if i % 100 == 0:
                    im = alg.vtable()
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    p = ax.imshow(im.transpose(), cmap=cm.jet, interpolation='nearest', vmin=0, vmax=10)
                    plt.colorbar(p)
                    fig.savefig(str(i)+'.png')
                    print str(i)

            # score environment
            cumrew += r
            output[j].append(cumrew)

            # next state
            s = sp


    # output in the appropriate format
    for i in range(env.len()):
        for j in range(num_avg):
            f.write(str(output[j][i]) + ',')
        f.write('\n')
    f.close()


#epsilons = [0, 0.001]
#aname = ['m','p']
epsilons = [0.001]
aname = ['p']

#k = [10, 20, 50, 100]
#k = [10, 20]
k = [200]

alphas = [-23] 

#envs = [BlockingWorld(), ShortcutWorld()]
#ename = ['bl', 'sc']
#envs = [AlternatingWorld()]
#ename = ['al']
envs = [ShortcutWorld()]
ename = ['sc']

maxlen = len(envs)*len(alphas)*len(k)
widgets = [pb.Bar('>'), ' ', pb.ETA(), ' ', pb.ReverseBar('<')]
pbar = pb.ProgressBar(widgets=widgets, maxval=maxlen).start()

count = 0
for x in range(len(envs)):
    for y in range(len(alphas)):
        for z in k:
            #filename = 'dyna_' + aname[y] + '_' + str(z) + '_' + ename[x] + '.txt'
            #eval(filename, envs[x], epsilons[y], z, 0)
            filename = 'pdyna_' + str(alphas[y]) + '_' + str(z) + '_' + ename[x] + '.txt'
            eval(filename, envs[x], 0, z, alphas[y])

            count += 1
            pbar.update(count)

pbar.finish()
