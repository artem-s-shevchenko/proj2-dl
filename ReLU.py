from torch import FloatTensor
from torch import LongTensor

from Module import *

class ReLU(Module):
    """
    Class of ReLU activation function module
    """
    def forward(self, x_in):
        """
        Forward pass of module

        :param x_in: input tensor
        :return: result of forward pass
        """
        self.last_input = x_in
        #easy way to implement: just clone and zero cells where values <= 0 
        temp = x_in.clone()
        temp[x_in <= 0] = 0 
        return temp
    def backward(self, dl_dx_out):
        """
        Backward pass of module

        :param dl_dx_out: tensor containing the gradient of the loss with respect to the module’s output
        :return:  tensor containing the gradient of the loss wrt the module’s input
        """
    	#since derivative consists of 0 and 1 and then we multiply it by dl_dx_out, it
    	#makes sense just to clone dl_dx_out and to zero cells where values of input were <= 0 
        temp = dl_dx_out.clone()
        temp[self.last_input <= 0] = 0 
        return temp					   
