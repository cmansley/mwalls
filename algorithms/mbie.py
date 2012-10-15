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

class MBIE:
    def __init__(self, delta):
        self.Q = collections.defaultdict(int)
        self.T = collections.defaultdict(DiceLearner)
        self.R = collections.defaultdict(DiceLearner)
        self.n = collections.defaultdict(self.getTime)
        self.time = 0
        
        self.states = set()

        # algorithm parameters
        self.delta = delta

        # environment parameter (optimization parameter)
        self.gamma = 0.9

    def reset(self):
        """Competely reset algorthim"""
        self.__init__(self.delta)

    def name(self):
        """Generate algorithm name with parameters"""
        return '_'.join(['mbie', str(self.delta)])

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

    def dot(T,V):
        """Compute dot product of two lists of numbers"""
        return sum([T[i]*V[i] for i in range(len(V))])

    def constrain_max(T, V, eps):
        """Compute maximum value of a distribution T given some constraints"""
        if 1-T[V.index(max(V))] <= eps/2:
            T_temp = [0 for i in T]
            T_temp[V.index(max(V))] = 1
            return (max(V), T_temp)
        else:
            T_temp = list(T)
            s_max = V.index(max(V))

            weight = eps/2.0
            T_temp[s_max] += weight
            v_sort = sorted([(v,i) for i,v in enumerate(V)])
            for v,i in v_sort:
                weight -= T_temp[i]
                T_temp[i] = 0.0
                if weight < 0:
                    T_temp[i] += math.fabs(weight)
                    break
            
        return (self.dot(T_temp,V), T_temp)

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

        thresh = 0.1

        delta = 1
        while delta > thresh:
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
                        elif sa not in self.n or (time-self.n[sa]) > self.dynam:
                            self.Q[sa] = rmax + self.gamma*vmax
                        else:
                            Vcop = [self.e(sp) for sp in T.keys()]
                            Tcop = [T[sp] for sp in T.keys()]
                            # add smax state
                            Tcop.append(0.0)
                            Vcop.append(vmax)
                            # compute optimal bounds
                            Rupper = math.sqrt(math.log(2/self.delta)*rmax*rmax / 2*self.R[sa].n)
                            # log_b(x) = log_k(x) / log_k(b)
                            # log_e(2^S - 2) = log_2(2^S - 2) / log_2(e)
                            # S*log_2(2)/log_2(e)
                            S = 400
                            eps = math.sqrt( 2*(S/math.log(math.e,2) - math.log(self.delta))/self.T[sa].n )

                            # compute backup with constraints
                            self.Q[sa] = self.R[sa].expectation() + Rupper + self.gamma*self.constrain_max(Tcop, Vcop, eps)

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
        
