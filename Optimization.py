from torch import FloatTensor
from torch import LongTensor

class Optimization(object):
    """
	Base class of optimization algorithm
    """
    def step(self, *input):
        """
        Step of optimization (stub function)

        :param input: possible parameters for step function
        """
        raise NonImplementedError
