
"""Sweet docstring for this module

Other things here
"""
from __future__ import division

import math
import random

class CoinLearner:
    """Sweet comment about this class.
    
    Define what is going on here.
    """
    def __init__(self, epsilon, delta):
        self.delta = delta
        self.epsilon = epsilon

        self.p = 0
        self.c = 0
        self.m = 1/(2*epsilon*epsilon)*math.log(2/delta)        
        
    def learn(self, obs):
        """Very similar in design to the KWIK coin-learner from Li 2010"""
        self.c += 1
        self.p = self.p + (obs-self.p)/self.c

    def predict(self):
        """None takes the place of an IDK"""
        if self.c < self.m:
            return None

        return self.p

    def interval(self):

        epsilon = math.sqrt(1/(2*self.c)*math.log(1/self.delta))
        a = self.p - epsilon
        b = self.p + epsilon

        return (a,b)


if __name__ == '__main__':
    
    epsilon = 0.1
    delta = 0.1

    cl = CoinLearner(epsilon, delta)
    
    for i in range(10):
    
        r = random.random()

        if r < 0.5:
            bit = 1
        else:
            bit = 0

        cl.learn(bit)

    # if abs(cl.predict() - 0.5) > epsilon:
    #     print "Epsilon Failure"
    # else:
    #     print "Epsilon Success"

    print cl.interval()
