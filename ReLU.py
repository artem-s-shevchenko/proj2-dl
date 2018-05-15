from torch import FloatTensor
from torch import LongTensor

from Module import *

class ReLU(Module):
    """Class of Relu activation function
    """
    def forward(self, x_in): 
        self.last_input = x_in
        #easy way to implement: just clone and zero cells where values <= 0 
        temp = x_in.clone()
        temp[x_in <= 0] = 0 
        return temp
    def backward(self, dl_dx_out):
    	#since derivative consists of 0 and 1 and then we multiply it by dl_dx_out, it
    	#makes sense just to clone dl_dx_out and to zero cells where values of input were <= 0 
        temp = dl_dx_out.clone()
        temp[self.last_input <= 0] = 0 
        return temp					   
