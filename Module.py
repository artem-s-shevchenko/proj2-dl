from torch import FloatTensor
from torch import LongTensor 

class Module(object):
    """
    Base class of module, from which the other classes of modules are inherited
    """
    def forward(self, *input): 
    	"""
        Forward pass of module (stub function)

        :param input: input tensors
        """
        raise NotImplementedError
    def backward(self, *gradwrtoutput): 
    	"""
        Backward pass of module (stub function)

        :param gradwrtoutput: tensors containing the gradient of the loss with respect to the module’s output
        """
        raise NotImplementedError
    def param(self):
    	"""
        Parameters of module (stub function)

        :return: empty list
        """
        return []