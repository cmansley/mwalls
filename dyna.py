#!/usr/bin/env python

from __future__ import division

import collections
import random
import math
import numpy
from dice import DiceLearner
from memory import MemoryLearner

class DynaQ:
    def __init__(self, k, epsilon):
        self.Q = collections.defaultdict(int)
        self.T = collections.defaultdict(DiceLearner)
        self.R = collections.defaultdict(DiceLearner)
        self.n = collections.defaultdict(self.getTime)
        self.time = 0
        
        self.states = set()

        # algorithm parameters
        self.beta = 0.5
        self.k = k
        self.epsilon = epsilon

        # environmental parameters
        self.gamma = 0.9

    def reset(self):
        """Competely reset algorthim"""
        self.__init__(self.k, self.epsilon)

    def name(self):
        """Generate algorithm name with parameters"""
        return '_'.join(['dyna', str(self.beta), str(self.k), str(self.epsilon)])

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
        self.Q[sa] = self.Q[sa] + self.beta*(reward + self.epsilon * math.sqrt(self.time - self.n[sa]) + self.gamma*self.e(sprime) - self.Q[sa]) 

        # update model
        self.T[sa].learn(sprime)
        self.R[sa].learn(reward)

        # simulated updates to Q "hypothetical experience"
        for i in range(self.k):
            # grab previously experienced state with a randomized action
            rs = random.choice(list(self.states))
            ra = random.randrange(4)
            rsa = (rs, ra)

            # simulate with model
            rsp = self.T[rsa].predict()
            rr = self.R[rsa].predict()

            # default behavior from Suttonq
            if rsp == None or rr == None:
                rsp = rs
                rr = 0

            # update evaluation function
            self.Q[rsa] = self.Q[rsa] + self.beta*(rr + self.epsilon * math.sqrt(self.time - self.n[rsa]) + self.gamma*self.e(rsp) - self.Q[rsa]) 

        
    def policy(self, state):
        """Take the maximum Q-valued action given the state"""
        #+ self.epsilon*math.sqrt(self.n[(state, i)])
        value, action = self._qmax(state)
        return action
