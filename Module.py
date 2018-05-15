from torch import FloatTensor
from torch import LongTensor 

class Module(object):
    """Base class of module, from which the other classes of modules are inherited
    """
    def forward(self, *input): 
        raise NotImplementedError
    def backward(self, *gradwrtoutput): 
        raise NotImplementedError
    def param(self): 
        return []