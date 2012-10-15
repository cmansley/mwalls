
"""Sweet docstring for this module

Other things here
"""
from __future__ import division

class MemoryLearner:
    """Sweet comment about this class.
    
    Define what is going on here.
    """
    def __init__(self):
        self.obs = None

    def learn(self, obs):
        """Simply memorize the obs """
        self.obs = obs

    def distribution(self):
        return {self.obs : 1}

    def predict(self):
        """ """
        return self.obs
    
    def expectation(self):
        """ Expectation is an element"""
        return self.obs
