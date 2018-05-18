from torch import FloatTensor
from torch import LongTensor

from Module import *

class Tanh(Module):
    """
    Class of tanh activation function module
    """
    def forward(self, x_in):
        """
        Forward pass of module

        :param x_in: input tensor
        :return: result of forward pass
        """
        self.last_input = x_in
        return x_in.tanh()
    def backward(self, dl_dx_out): 
        """
        Backward pass of module

        :param dl_dx_out: tensor containing the gradient of the loss with respect to the module’s output
        :return:  tensor containing the gradient of the loss wrt the module’s input
        """
        return (1-self.last_input.tanh().pow(2))*dl_dx_out
