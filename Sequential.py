from torch import FloatTensor
from torch import LongTensor 

from Module import *


class Sequential (Module):
    def __init__(self, *modules):
        """
        Class of Sequential module, it combines other modules into one model

        :param modules: module or modules to be used in model
        """
        super(Sequential, self).__init__()
        self.modules = modules
    def forward(self, x_in):
        """
        Forward pass of model

        :param x_in: input tensor
        :return: result of forward pass
        """
        for module in self.modules:
            x_in = module.forward(x_in)
        x_out = x_in
        return x_out
    def backward(self, dl_dx_out): 
        """
        Backward pass of module

        :param dl_dx_out: tensor containing the gradient of the loss with respect to the output of the last layer of model
        """
        for module in reversed(self.modules):
            dl_dx_out = module.backward(dl_dx_out)
    def param(self): 
        """
        Parameters of model

        :return: return a list of pairs, each composed of a parameter tensor and a gradient tensor of same size
        """
        parameters = []
        for module in self.modules:
            parameters.extend(module.param())
        return parameters


