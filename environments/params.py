class Params():
    def __init__(self):
        self.length = 1000
        self.cycle = 1000
        self.transition = 500
        self.probability = -1

    def name(self):
        return str(self.length)+'_'+str(self.cycle)+'_'+str(self.transition)+'_'+str(self.probability)
