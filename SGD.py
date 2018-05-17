from torch import FloatTensor
from torch import LongTensor

from Optimization import *

class SGD(Optimization):
    def __init__(self, model, lr, momentum=0):
        super(SGD, self).__init__()
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.momentum_buffer = {}
    def step(self):
        for p in self.model.param():
            if p not in self.momentum_buffer.keys():
                self.momentum_buffer[p] = FloatTensor(p[0].size()).zero_()
            self.momentum_buffer[p] = self.momentum_buffer[p].mul_(self.momentum).add_(self.lr*p[1])
            p[0].sub_(self.momentum_buffer[p])