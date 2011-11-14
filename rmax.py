#!/usr/bin/env python

from __future__ import division

import collections
import random
import math
import numpy
import itertools

from dice import DiceLearner
from memory import MemoryLearner
from powerdice import PowerDiceLearner

class Rmax:
    def __init__(self):
        self.Q = collections.defaultdict(int)
        self.T = collections.defaultdict(MemoryLearner)
        self.R = collections.defaultdict(MemoryLearner)
        self.n = collections.defaultdict(self.getTime)
        self.time = 0
        
        self.states = set()

        # environment parameter (optimization parameter)
        self.gamma = 0.9

    def getTime(self):
        return self.time

    def _qmax(self,state):
        q =  [ self.Q[(state, action)] for action in range(4) ] 

        maxq = max(q)
        actions = [ i for i in range(4) if q[i] == maxq ] 

        action = random.choice(actions)
        value = q[action]

        return (value, action)

    def e(self, state):
        """Take the max of the Q-values given this state by iterating over actions"""
        value, action = self._qmax(state)
        return value

    def learn(self, state, action, reward, sprime):
        """Add real experience to model, then simulate experience with model"""

        self.time += 1
        
        # state and action tuple
        sa = (state, action)

        # keep track of experienced states
        self.states.add(state)

        # reset/increment exploration bonus count
        self.n[sa] = self.time

        # update model
        self.T[sa].learn(sprime)
        self.R[sa].learn(reward)

    def solve(self,time):
        """Solve model using value iteration"""

        rmax = 1
        vmax = 1/(1-self.gamma)

        theta = 0.1

        delta = 1
        while delta > theta:
            delta = 0
            for s in self.states:
                v = self.e(s)
                for a in range(4):
                    sa = (s, a)
                    #if self.T[sa].n < 3 or (time-self.n[sa]) > 1500:
                    if sa not in self.T:
                        self.Q[sa] = rmax + self.gamma*vmax
                    else:
                        T = self.T[sa].distribution()
                        if not T:
                            self.Q[sa] = rmax + self.gamma*vmax
                        elif sa not in self.n or (time-self.n[sa]) > 1000:
                            self.Q[sa] = rmax + self.gamma*vmax
                        else:
                            values = [T[sp]*self.e(sp) for sp in T.keys()]
                            self.Q[sa] = self.R[sa].expectation() + self.gamma*sum(values)



                delta = max([delta, math.fabs(v-self.e(s))])

    def policy(self, state, time):
        """Take the maximum Q-valued action given the state"""
        self.solve(time)
        value, action = self._qmax(state)
        return action    

    def vtable(self):
        im = numpy.ones((9,6))
        for state in itertools.product(range(9), range(6)):
            im[state[0]][state[1]] = self.e(state)
        return im

    def _maxvisit(self,state,time):
        to =  [ time - self.n[(state, action)] for action in range(4) ] 

        maxq = min(to)
        actions = [ i for i in range(4) if to[i] == maxq ] 

        action = random.choice(actions)
        value = to[action]

        return value

    def visits(self, time):
        im = numpy.ones((9,6))
        for state in itertools.product(range(9), range(6)):
            im[state[0]][state[1]] = self._maxvisit(state, time)
        return im
        
