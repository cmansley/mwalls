
"""Sweet docstring for this module

Other things here
"""
from __future__ import division

import collections
import random
import numpy as np

class DiceLearner:
    """Sweet comment about this class.
    
    Define what is going on here.
    """
    def __init__(self):
        self.t = collections.defaultdict(int)
        self.n = 0
        
    def learn(self, obs):
        """Very similar in design to the KWIK dice-learner from Li 2010"""
        self.n += 1

        alpha = 1/self.n

        for key in self.t.keys():
            self.t[key] = (1-alpha)*self.t[key]

        self.t[obs] = self.t[obs] + alpha

    def distribution(self):
        return t

    def predict(self):
        """Generate a random pull from the multinomial

        This should return exactly what was passed in. So, if the
        input is states, then the return values will be states.
        """
        if self.n == 0:
            return None
        
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

    dl = DiceLearner()
    
    for i in range(1000):
    
        r = random.randint(1,6)

        dl.learn(r)

    if abs(dl.expectation() - 3.5) > 0.3:
        print "Expectation Failure"
    else:
        print "Expecation Success"
    
