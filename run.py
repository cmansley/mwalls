#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import progressbar as pb

# worlds 
from environments.blocking import BlockingWorld
from environments.shortcut import ShortcutWorld
from environments.alternating import AlternatingWorld
from environments.geometricworld import GeometricWorld

# algorithms
from algorithms.dyna import DynaQ
from algorithms.powerdyna import PowerDynaQ
from algorithms.rmax import Rmax

num_avg = 10

def eval(outfile, alg, env):

    f = open(outfile, "w")

    output = []

    for j in range(num_avg):

        # reset environment/scores
        cumrew = 0.0
        s = env.ss()
        alg.reset()
  
        output.append([])

        # execute environment
        for i in range(env.len()):
            # grab action
            a = alg.policy(s)

            # grab next state
            sp, r = env.step(s,a,i)

            # pass next state to learner
            alg.learn(s,a,r,sp)

            if i < 5000 and False:
                if i % 100 == 0:
                    im = alg.vtable()
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    p = ax.imshow(im.transpose(), cmap=cm.jet, interpolation='nearest', vmin=0, vmax=10)
                    plt.colorbar(p)
                    fig.savefig(str(i)+'.png')
                    print str(i)

            if i < 5000 and False:
                if i % 100 == 0:
                    im = alg.visits(i)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    p = ax.imshow(im.transpose(), cmap=cm.jet, interpolation='nearest')
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

def main():
    # params = [0, 0.001]
    # aname = ['m','p']

    # params = [0.001]
    # params = [-23] 
    params = [9000, 10000]

    # backups = [10, 20, 50, 100]
    # backups = [10, 20]
    backups = [100]

    # envs = [BlockingWorld(), ShortcutWorld()]
    # envs = [AlternatingWorld()]
    # envs = [ShortcutWorld()]
    envs = [GeometricWorld()]

    maxlen = len(envs)*len(backups)*len(params)
    widgets = [pb.Bar('>'), ' ', pb.ETA(), ' ', pb.ReverseBar('<')]
    pbar = pb.ProgressBar(widgets=widgets, maxval=maxlen).start()

    count = 0
    for env in envs:
        for k in backups:
            for param in params:
                # alg = PowerDynaQ(k, param)
                # alg = DynaQ(k, param)
                alg = Rmax(1, param)

                filename = alg.name() + '_' + env.name() + '.txt'
                eval(filename, alg, env)

                count += 1
                pbar.update(count)

    pbar.finish()

if __name__ == '__main__':
    main()
