from torch import FloatTensor
from torch import LongTensor

from Module import *

class Tanh(Module):
    """Class of Tanh activation function
    """
    def forward(self, x_in):
        self.last_input = x_in
        return x_in.tanh()
    def backward(self, dl_dx_out): 
        return (1-self.last_input.tanh().pow(2))*dl_dx_out
