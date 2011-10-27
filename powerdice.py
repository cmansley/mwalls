
"""Sweet docstring for this module

Other things here
"""
from __future__ import division

import collections
import random
import numpy as np
import math

class PowerDiceLearner:
    """Sweet comment about this class.
    
    Define what is going on here.
    """
    def __init__(self):
        self.obs = []
        self.t = collections.defaultdict(int)
        self.gamma = -0.3
        self.n = 0
        
    def learn(self, obs, time):
        """ """
        self.obs.append((time, obs))

    def distribution(self):
        """ Doesn't work"""
        return self.t

    def predict(self, time):
        """Generate a random pull from the multinomial

        This should return exactly what was passed in. So, if the
        input is states, then the return values will be states.
        """
        if not self.obs:
            return None

        # compute kernel weight average 
        self.t = collections.defaultdict(int)
        norm = 0
        for t, obs in self.obs:
            dis = math.pow(time - t, self.gamma)
            self.t[obs] += dis
            norm += dis

        for key in self.t.keys():
            self.t[key] = self.t[key] / norm
            

        # draw transition 
        values = self.t.values();
        keys = self.t.keys();
        
        temp = np.random.multinomial(1, values)
        i = np.argmax(temp)

        return keys[i]
    
    def expectation(self):
        """Compute expectation of dice learner
        
        This may not be logical for transition probabilities, but it
        should be fine for reward functions or other scaler valued
        functions
        """
        e = sum( [key*self.t[key] for key in self.t.keys()] )

        return e



if __name__ == '__main__':

    dl = PowerDiceLearner()
    
#    for i in range(1000):
    
#        r = random.randint(1,6)

#        dl.learn(r, i)

    dl.learn(3, 0)
    dl.learn(6, 999)

    dl.predict(1000)
    print dl.expectation()
    print dl.distribution()

    if abs(dl.expectation() - 3.5) > 0.3:
        print "Expectation Failure"
    else:
        print "Expecation Success"
    
