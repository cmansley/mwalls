import random

# Grid World Environment
class GeometricWorld():
    """Cool description
    
    """
    def __init__(self):
        """ Constructor for grid world """
        
        # environment params
        self.size = (8, 5)  # max grid location
        self.goal = (8, 5)
        self.start = (3, 0)

        self.toggle = True

    def ss(self):
        return self.start

    def name(self):
        """Report environment name"""
        return 'gw'

    def len(self):
        return 8000

    def step(self, state, action, time):
        """Next state """

        flip = random.random()
        
        if flip < 0.001:
            self.toggle = not self.toggle

        x,y = state

        # Action logic
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

        # barrier
        # from sutton!!!
        if self.toggle:
            if state[1] == 1 and action == 0:
                if state[0] != 8:
                    y = state[1]

            if state[1] == 3 and action == 2:
                if state[0] != 8:
                    y = state[1]
        else:
            if state[1] == 1 and action == 0:
                if state[0] != 0 and state[0] != 8:
                    y = state[1]

            if state[1] == 3 and action == 2:
                if state[0] != 0 and state[0] != 8:
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

