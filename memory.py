
"""Sweet docstring for this module

Other things here
"""
from __future__ import division

class MemoryLearner:
    """Sweet comment about this class.
    
    Define what is going on here.
    """
    def __init__(self):
        self.store = None
        self.t = {}

    def learn(self, obs):
        """Simply memorize the obs """

        self.store = obs
        self.t[obs] = 1

    def distribution(self):
        return t

    def predict(self):
        """ """
        return self.store
    
    def expectation(self):
        return obs
