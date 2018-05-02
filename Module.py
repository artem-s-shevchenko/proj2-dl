from torch import FloatTensor
from torch import LongTensor 

epsilon = 1e-6

class Module(object):
    def forward(self, *input): 
        raise NotImplementedError
    def backward(self, *gradwrtoutput): 
        raise NotImplementedError
    def param(self): 
        return []