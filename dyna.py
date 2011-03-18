#!/usr/bin/env python

from __future__ import division

import collections
import random
import math
import numpy
from dice import DiceLearner

class DynaQ:
    def __init__(self):
        self.Q = collections.defaultdict(int)
        self.T = collections.defaultdict(DiceLearner)
        self.R = collections.defaultdict(DiceLearner)
        self.n = collections.defaultdict(int)
        
        self.beta = 0.5
        self.gamma = 0.9
        self.k = 10
        self.epsilon = 0.001

    def _qmax(self,state):
        q =  [ self.Q[(state, action)] for action in range(4) ] 

        maxq = -100
        actions = []
        for i in range(4):
            if q[i] > maxq:
                maxq = q[i]
                actions = []
                actions.append(i)
            elif q[i] == maxq:
                actions.append(i)
            else:
                pass

        action = random.choice(actions)
        value = q[action]

        return (value, action)

    def e(self, state):
        """Take the max of the Q-values given this state by iterating over actions"""
        value, action = self._qmax(state)
        return value

    def learn(self, state, action, reward, sprime):
        """Add real experience to model, then simulate experience with model"""
        
        sa = (state, action)

        # reset/increment exploration bonus count
        self.n[sa] = -1
        for key in self.n.keys():
            self.n[key] += 1 

        # update evaluation/policy functon
        self.Q[sa] = self.Q[sa] + self.beta*(reward + self.epsilon * math.sqrt(self.n[sa]) + self.gamma*self.e(sprime) - self.Q[sa]) 

        # update model
        self.T[sa].learn(sprime)
        self.R[sa].learn(reward)

        # simulated updates to Q "hypothetical experience"
        for i in range(self.k):
            # grab previously experienced state with a randomized action
            rs, ra = random.choice(self.T.keys())
            ra = random.randrange(4)
            rsa = (rs, ra)

            # simulate with model
            rsp = self.T[rsa].predict()
            rr = self.R[rsa].predict()

            # default behavior from Suttonq
            if rsp == None or rr == None:
                rsp = rsa[0]
                rr = 0

            # update evaluation function
            self.Q[rsa] = self.Q[rsa] + self.beta*(rr + self.epsilon * math.sqrt(self.n[rsa]) + self.gamma*self.e(rsp) - self.Q[rsa]) 

        
    def policy(self, state):
        """Take the maximum Q-valued action given the state"""
        value, action = self._qmax(state)
        return action
