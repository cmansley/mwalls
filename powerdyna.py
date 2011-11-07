#!/usr/bin/env python

from __future__ import division

import collections
import random
import math
import numpy
from dice import DiceLearner
from memory import MemoryLearner
from powerdice import PowerDiceLearner

class PowerDynaQ:
    def __init__(self, k):
        self.Q = collections.defaultdict(int)
        self.T = collections.defaultdict(self.factory)
        self.R = collections.defaultdict(self.factory)
        self.n = collections.defaultdict(self.getTime)
        self.time = 0
        
        self.states = set()

        self.beta = 0.5
        self.gamma = 0.9
        self.k = k
        
        self.alpha = -4

    def factory(self):
        return PowerDiceLearner(self.alpha)

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
        
        sa = (state, action)

        # keep track of experienced states
        self.states.add(state)

        # reset/increment exploration bonus count
        self.n[sa] = self.time

        # update evaluation/policy functon
        self.Q[sa] = self.Q[sa] + self.beta*(reward + self.gamma*self.e(sprime) - self.Q[sa]) 

        # update model
        self.T[sa].learn(sprime, self.time)
        self.R[sa].learn(reward, self.time)

        # simulated updates to Q "hypothetical experience"
        for i in range(self.k):
            # grab previously experienced state with a randomized action
            rs = random.choice(list(self.states))
            ra = random.randrange(4)
            rsa = (rs, ra)

            # simulate with model
            rsp = self.T[rsa].predict(self.time+1)
            rr = self.R[rsa].predict(self.time+1)

            # default behavior from Suttonq
            if rsp == None or rr == None:
                rsp = rs
                rr = 0

            # rmax 
            if rsp == -1:
                rmax = 0
                vmax = rmax / (1 - self.gamma)
                rr = rmax
                self.Q[rsa] = self.Q[rsa] + self.beta*(rr + self.gamma*vmax - self.Q[rsa]) 
            else:
                # update evaluation function
                self.Q[rsa] = self.Q[rsa] + self.beta*(rr + self.gamma*self.e(rsp) - self.Q[rsa]) 

        
    def policy(self, state):
        """Take the maximum Q-valued action given the state"""
        value, action = self._qmax(state)
        return action
