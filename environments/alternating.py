# Grid World Environment
class AlternatingWorld():
    """Cool description
    
    """
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3

    def __init__(self):
        """ Constructor for grid world """
        
        # environment params
        self.size = (8, 5)  # max grid location
        self.goal = (8, 5)
        self.start = (4, 0)

    def ss(self):
        return self.start

    def name(self):
        """Report environment name"""
        return 'al'

    def len(self):
        return 12000

    def step(self, state, action, time):
        """Next state """

        x,y = state

        # Action logic
        if action == AlternatingWorld.UP:
            y += 1
        elif action == AlternatingWorld.LEFT:
            x += 1
        elif action == AlternatingWorld.DOWN:
            y -= 1
        elif action == AlternatingWorld.RIGHT:
            x -= 1
        else:
            raise Exception("Hello")

        # barrier
        # from sutton!!!
        if time % 6000 < 3000 :
            if state[1] == 1 and action == AlternatingWorld.UP:
                if state[0] != 0:
                    y = state[1]

            if state[1] == 2 and action == AlternatingWorld.DOWN:
                if state[0] != 0:
                    y = state[1]
        else:
            if state[1] == 1 and action == AlternatingWorld.UP:
                if state[0] != 0 and state[0] != 8:
                    y = state[1]

            if state[1] == 2 and action == AlternatingWorld.DOWN:
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

