from torch import FloatTensor
from torch import LongTensor

from Module import *

class LossMSE(Module):
    """
    Class of MSE loss module
    """
    def forward(self, x_in, target):
        """
        Forward pass of module

        :param x_in: input tensor
        :param target: target tensor
        :return: result of forward pass
        """
        self.last_input = x_in
        self.last_target = target
        return (x_in.sub(target)).pow(2).mean()
    def backward(self):
        """
        Backward pass of module

        :return:  tensor containing the gradient of the loss wrt the input
        """
        return 2*(self.last_input-self.last_target)/(self.last_target.size()[0]*self.last_target.size()[1])