from torch import FloatTensor
from torch import LongTensor 

from Linear import *
from ReLU import *
from Tanh import *
from Module import *


class Sequential (Module):
    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.modules = modules
    def forward(self, x_in):
        for module in self.modules:
            x_in = module.forward(x_in)
        x_out = x_in
        return x_out
    def backward(self, dl_dx_out): 
        for module in reversed(self.modules):
            dl_dx_out = module.backward(dl_dx_out)
    def param(self): 
        parameters = []
        for module in self.modules:
            parameters.extend(module.param())
        return parameters


