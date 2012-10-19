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
    def __init__(self, m, dynam, alpha, beta):
        self.Q = collections.defaultdict(int)
        self.T = collections.defaultdict(MemoryLearner)
        self.R = collections.defaultdict(MemoryLearner)
        self.T_old = collections.defaultdict(MemoryLearner)
        # future work
        # self.R_old = collections.defaultdict(MemoryLearner)
        self.n = collections.defaultdict(self.getTime)
        self.interval = collections.defaultdict(lambda: dynam)
        self.time = 0
        
        self.states = set()

        # algorithm parameters
        self.m = m
        self.dynam = dynam
        self.alpha = alpha
        self.beta = beta

        # environment parameter (optimization parameter)
        self.gamma = 0.9

    def reset(self):
        """Competely reset algorthim"""
        self.__init__(self.m, self.dynam, self.alpha, self.beta)

    def name(self):
        """Generate algorithm name with parameters"""
        return '_'.join(['rmax', str(self.m), str(self.dynam), str(self.alpha), str(self.beta)])

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
        rmin = 0
        vmax = 1/(1-self.gamma)

        theta = 0.1

        delta = 1
        while delta > theta:
            delta = 0
            for s in self.states:
                v = self.e(s)
                for a in range(4):
                    sa = (s, a)
                    if sa not in self.T:
                        self.Q[sa] = rmax + self.gamma*vmax
                    else:
                        T = self.T[sa].distribution()
                        if not T:
                            self.Q[sa] = rmax + self.gamma*vmax
                        elif (time-self.n[sa]) > self.interval[sa]:                            
                            # compare old transition and this transition
                            if sa in self.T_old:
                                if self.T_old[sa].predict() == self.T[sa].predict():
                                    # optional algorithm
                                    self.interval[sa] = self.alpha*self.interval[sa] + self.beta

                            # store old pair
                            self.T_old[sa] = self.T[sa]

                            # reset state action pair
                            del self.T[sa]
                            del self.R[sa]

                            # do optimistic backup
                            self.Q[sa] = rmax + self.gamma*vmax
                        else:
                            values = [T[sp]*self.e(sp) for sp in T.keys()]
                            self.Q[sa] = self.R[sa].expectation() + self.gamma*sum(values)



                delta = max([delta, math.fabs(v-self.e(s))])

    def policy(self, state):
        """Take the maximum Q-valued action given the state"""
        self.solve(self.time)
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
        
