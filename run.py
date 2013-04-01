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
    # create algorithms
    algorithms = []

    backups = [10]
    # aname = ['m','p']
    params = [0, 0.001]
    for k in backups:
        for param in params:
            algorithms.append(DynaQ(k, param))

    # rmax intervals
    params = [160, 320, 640, 1280, 2560, 5120, 10240]

    # const
    constants = [50, 100, 200, 300, 400, 500]
    for k in constants:
        for param in params:
            algorithms.append(Rmax(1, param, 1, k))

    # exponential (includes constant/fixed)
    constants = [1, 1.5, 2.5, 3]
    for k in constants:
        for param in params:
            algorithms.append(Rmax(1, param, k, 0))


    # for k in backups:
    #     for param in params:
    #         algorithms.appendPowerDynaQ(k, param))

    # create environments
    # envs = [BlockingWorld(), ShortcutWorld()]
    envs = [AlternatingWorld()]
    # envs = [ShortcutWorld()]
    # envs = [GeometricWorld()]

    maxlen = len(envs)*len(algorithms)
    widgets = [pb.Bar('>'), ' ', pb.ETA(), ' ', pb.ReverseBar('<')]
    pbar = pb.ProgressBar(widgets=widgets, maxval=maxlen).start()

    count = 0
    for env in envs:
        for alg in algorithms:
            filename = alg.name() + '_' + env.name() + '.txt'
            eval(filename, alg, env)

            count += 1
            pbar.update(count)

    pbar.finish()

if __name__ == '__main__':
    main()
