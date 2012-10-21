from __future__ import division

import sys
import re
import math

import matplotlib.pyplot as plt

def stats(data):
    m = sum(tempdata)/len(tempdata)

    s = math.sqrt(sum([(x-m)*(x-m) for x in data]) / (len(data) - 1))

    u = m+s
    l = m-s
    
    return (m, u, l)

fig, ax = plt.subplots(1)
fig2, ax2 = plt.subplots(1)

# for each file on the command line
for arg in sys.argv[1:]:

    mean = []
    up = []
    low = []

    # open the file
    with open(arg, 'r') as f:

        # for each line
        i = 0
        n = []
        for line in f:
            # parse line (might be dumb)
            tempdata = [float(x) for x in re.split('[ ,]+',line.strip('\n,'))]

            # add line data
            m,u,l = stats(tempdata)
            mean.append(m)
            up.append(u)
            low.append(l)
            n.append(i)
            i += 1

        ax.plot(n, mean, label=arg)
        #ax.fill_between(n, up, low, alpha=0.5)
        
        s = 50
        slope = [(mean[i+s] - mean[i])/s for i in range(len(mean)-s)]
        ax2.plot(n[:-s], slope, label=arg)

ax.set_xlabel('Time/Experience')
ax.set_ylabel('Cumulative Reward')
ax.set_ybound(lower=0, upper=800)
ax.legend(loc='upper left')
fig.savefig('output.png')

ax2.legend(loc='upper left')
fig2.savefig('slope.png')

