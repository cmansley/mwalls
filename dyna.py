#!/usr/bin/env python

from __future__ import division

import collections
import random
from dice import DiceLearner

class DynaQ:
    def __init__(self):
        self.Q = collections.defaultdict(int)
        self.T = collections.defaultdict(DiceLearner)
        self.R = collections.defaultdict(DiceLearner)
        
        self.beta = 0.5
        self.gamma = 0.9
        self.k = 10

    def _qmax(self,state):
        return max( [ (self.Q[(state, action)], action) for action in range(4) ] )

    def e(self, state):
        """Take the max of the Q-values given this state by iterating over actions"""
        value, action = self._qmax(state)
        return value

    def learn(self, state, action, reward, sprime):
        """Add real experience to model, then simulate experience with model"""

        # update evaluation/policy functon
        sa = (state, action)
        self.Q[sa] = self.Q[sa] + self.beta*(reward + self.gamma*self.e(sprime) - self.Q[sa]) 

        # update model
        self.T[sa].learn(sprime)
        self.R[sa].learn(reward)

        # simulated updates to Q "hypothetical experience"
        for i in range(self.k):
            # grab previously experienced state-action pair
            rsa = random.choice(self.T.keys())

            # simulate with model
            rsp = self.T[rsa].predict()
            rr = self.R[rsa].predict()

            # update evaluation function
            self.Q[rsa] = self.Q[rsa] + self.beta*(rr + self.gamma*self.e(rsp) - self.Q[rsa]) 

        
    def policy(self, state):
        """Take the maximum Q-valued action given the state"""
        value, action = self._qmax(state)
        return action
