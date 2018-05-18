from torch import FloatTensor
from torch import LongTensor

class Optimization(object):
    """
	Base class of optimization algorithm
    """
    def step(self):
        """
        Step of optimization (stub function)
        """
        raise NonImplementedError
