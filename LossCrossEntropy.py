from torch import FloatTensor
from torch import LongTensor

from Module import *

class LossCrossEntropy(Module):
    """
    Class of CrossEntropy loss module
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
        temp1 = x_in.gather(1, target.view(-1,1)).exp().squeeze()
        temp2 = x_in.exp().sum(1)
        return (-1.0/x_in.size()[0])*((temp1/temp2).log().sum())
    def backward(self):
        """
        Backward pass of module

        :return: tensor containing the gradient of the loss wrt the input
        """
        temp1 = self.last_input.exp().sum(1).view(-1,1)
        temp2 = -1*self.last_input.exp()/temp1
        j = LongTensor(list(range(self.last_input.size(0))))
        temp2[j, self.last_target] = temp2[j, self.last_target] + 1
        return (-1.0/self.last_input.size()[0])*temp2
