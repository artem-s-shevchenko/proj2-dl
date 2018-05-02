from torch import FloatTensor
from torch import LongTensor

from Module import *

class ReLU(Module):
    def forward(self, x_in): 
        self.last_input = x_in
        temp = x_in.clone()
        temp[x_in <= 0] = 0
        return temp
    def backward(self, dl_dx_out):
        temp = dl_dx_out.clone()
        temp[self.last_input <= 0] = 0
        return temp
