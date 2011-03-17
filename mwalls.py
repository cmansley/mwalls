#!/usr/bin/env python

from __future__ import division

import itertools
import collections
import random

# Iterator wrapper for states
class StateIterator:
    def __init__(self,x,y):
        self._iter = itertools.product(range(x),range(y))

    def __iter__(self):
        self._iter.__iter__()
        return self

    def next(self):
        next = self._iter.next()
        return next

# Dice learner 
class DiceLearner:
    def __init__(self):
        self.t = collections.defaultdict(int)
        self.n = 0
        
    def learn(self, obs):
        self.n += 1

        alpha = 1/self.n

        for key in self.t.keys():
            self.t[key] = (1-alpha)*self.t[key]

        self.t[obs] = self.t[obs] + alpha

    def predict(self):
        return t


# Environment Superclass
class Environment:
    def __init__(self):
        """ Constructor for environment """
        pass

# Grid World Environment
class GridWorld(Environment):
    def __init__(self):
        """ Constructor for grid world """
        Environment.__init__(self)
        
        # environment params
        self.size = 15 # max grid location
        self.goal = (14, 8)
        self.start = (0, 4)

    def ss(self):
        return self.start

    def step(self, state, action):
        """ Next state """

        x,y = state

        # Action logic
        if action == 0:
            y += 1
        elif action == 1:
            x += 1
        elif action == 2:
            y -= 1
        elif action == 3:
            x -= 1
        else:
            raise Exception("Hello")

        # barrier
        if state[0] == 2 and action == 1:
            if state[1] != 15:
                x = state[0]
        
        # Wall logic
        if x > self.size or x < 0 or y > self.size or y < 0:
            sprime = state
        else:
            sprime = (x,y)
        
        # Goal logic
        if sprime == self.goal:
            reward = 1
            terminal = True
        else:
            reward = -0.1
            terminal = False

        return (sprime, reward, terminal)


class Learner:
    def __init__(self):
        pass

class BlackBox(Learner):
    def __init__(self):
        self.n = collections.defaultdict(int)
        self.T = collections.defaultdict(int)
        self.R = collections.defaultdict(int)
        self.V = collections.defaultdict(int)

    def learn(self, state, action, reward, sprime):
        """ Incorporate sars into the model """
        
        sas = (state, action, sprime)

        # update model 
        self.n[sas] += 1
        delta = reward - self.R[sas]
        self.R[sas] += delta/self.n[sas]

        for sp in StateIterator(16,16):
            sas = (state, action, sp)
            if sprime == sp:
                self.T[sas] = (1-alpha)*self.T[sas] + alpha
            else:
                self.T[sas] = (1-alpha)*self.T[sas]

    def predict(self, state, action):
        """ Makes a prediction based on the model """

        return (sprime, reward)

# Algorithm Superclass
class Algorithm:
    def __init__(self):
        """ Constructor for algorithm """
        pass

    def learn(self, state, action, reward, sprime):
        """ Incorporate sars data into learning algorithm """
        pass

    def policy(self, state):
        """ Generate action from internal representation"""
        pass
            

# Rmax Algorithm
class RMax(Algorithm):
    def __init__(self):
        """ Constructor for R_max """

        self.gamma = 0.9
        self.rmax = 1
        self.m = 1

        self.vmax = 1/(1-self.gamma)

    def _valueIteration(self):
        """ Value iteration on model """

        delta = 1

        # naive implementation
        while delta > 0.05:
            delta = 0
            for state in StateIterator(16,16):
                maxv = -1000
                v = self.V[state]
                for action in range(4):
                    temp = 0.0
                    normalizing = 0.0
                    for sprime in StateIterator(16,16):
                        sas = (state, action, sprime)

                        if self.T.has_key(sas):
                            temp += self.T[sas]*(self.R[sas]+self.gamma*self.V[sprime])
                            normalizing += self.T[sas]

                    if normalizing < self.m:
                        temp = self.vmax
                    else:
                        temp = temp / normalizing
                    
                    if temp > maxv:
                        maxv = temp

                self.V[state] = maxv

                delta = max(delta, abs(maxv - v))
        
    def policy(self, state):
        """ Generate action from internal representation """

        self._valueIteration()

        maxv = -100
        maxa = -1

        for action in range(4):
            temp = 0.0
            normalizing = 0.0
            for sprime in StateIterator(16,16):
                sas = (state, action, sprime)
                if self.T.has_key(sas):
                    temp += self.T[sas]*(self.R[sas]+self.gamma*self.V[sprime])
                    normalizing += self.T[sas]

            if normalizing < self.m:
                temp = self.vmax
            else:
                temp = temp / normalizing

            if temp > maxv:
                maxv = temp
                maxa = action

        return maxa


# Experimental Framework 

# averaging
# for i in range(number of trials to average over)

env = GridWorld()
alg = RMax()

# number of epsiodes
for i in range(40):

    # episode
    cumrew = 0.0
    s = env.ss()

    term = False
    while not term:
        
        a = alg.policy(s)
        
        sp, r, term = env.step(s,a)

        alg.learn(s,a,r,sp)

        cumrew += r

        s = sp

    print cumrew
