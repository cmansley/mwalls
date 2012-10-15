from params import Params
import random

# Grid World Environment
class GridWorld():
    """
    Parameterized environment supporting several types
    """
    def __init__(self, params):
        """Constructor for grid world"""
        
        # environment params
        # currently hard coded, maybe parameterized
        self.size = (8, 5)  # max grid location
        self.goal = (8, 5)
        self.start = (3, 0)

        # start in MDP 2
        self.toggle = False

        # store params
        self.params = params

    def ss(self):
        return self.start

    def name(self):
        """Report environment name"""
        return params.name()

    def len(self):
        return params.length

    def decide(self, time):
        if params.probability > 0:
            flip = random.random() 
            if flip < 0.001:
                self.toggle = not self.toggle
        else:
            self.toggle = time % params.transition < params.transition

    def step(self, state, action, time):
        """Next state """
        
        # decide our current MDP
        self.decide(time)

        # unpack state
        x,y = state

        # action logic
        if action == 0:
            y += 1
        elif action == 1:
            x += 1
        elif action == 2:
            y -= 1
        elif action == 3:
            x -= 1
        else:
            raise Exception("Hello")

        # decide which MDP 
        if self.toggle:
            # MDP 1
            if state[1] == 1 and action == 0:
                if state[0] != 8:
                    y = state[1]

            if state[1] == 3 and action == 2:
                if state[0] != 8:
                    y = state[1]
        else:
            # MDP 2
            if state[1] == 1 and action == 0:
                if state[0] != 0:
                    y = state[1]

            if state[1] == 3 and action == 2:
                if state[0] != 0:
                    y = state[1]

        # Wall logic
        if x > self.size[0] or x < 0 or y > self.size[1] or y < 0:
            sprime = state
        else:
            sprime = (x,y)
        
        # Goal logic
        if sprime == self.goal:
            reward = 1
            sprime = self.start
        else:
            reward = 0

        return (sprime, reward)

