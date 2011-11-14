
"""Sweet docstring for this module

Other things here
"""
from __future__ import division

class MemoryLearner:
    """Sweet comment about this class.
    
    Define what is going on here.
    """
    def __init__(self):
        self.t = {}

    def learn(self, obs):
        """Simply memorize the obs """
        self.t = {}
        self.t[obs] = 1

    def distribution(self):
        return self.t

    def predict(self):
        """ """
        return self.t.keys()[0]
    
    def expectation(self):
        """ Expectation is an element"""
        return self.t.keys()[0]
