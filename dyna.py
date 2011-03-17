#!/usr/bin/env python

from __future__ import division

import itertools
import collections
import random
import numpy.random as rand

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

    def distribution(self):
        return t

    def predict(self):
        pass
    
    def expectation(self):
        """
        Compute expectation of dice learner
        
        This may not be logical for transition probabilities, but it
        should be fine for reward functions or other scaler valued
        functions
        """
        sum( [key*self.t[key] for key in self.t.keys()] )

        return e


class DynaQ:
    def __init__(self):
        self.Q = collections.defaultdict(collections.defaultdict(int))
        self.beta = 0.5
        self.gamma = 0.9

    def _qmax(self,state):
        return max( [ (Q[(state, action)], action) for action in range(4) ] )

    def e(self, state):
        """Take the max of the Q-values given this state by iterating over actions"""
        value, action = self._qmax(state)
        return value

    def learn(self, state, action, reward, sprime):
        """Add real experience to model, then simulate experience with model"""

        sa = (state, action)
        Q[sa] = Q[sa] + self.beta*(reward + self.gamma*self.e(state) - Q[sa]) 
        
    def policy(self, state):
        """Take the maximum Q-valued action given the state"""
        value, action = self._qmax(state)
        return action
