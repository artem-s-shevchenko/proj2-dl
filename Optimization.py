from torch import FloatTensor
from torch import LongTensor

class Optimization(object):
    def step(self):
        raise NonImplementedError
