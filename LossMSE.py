from torch import FloatTensor
from torch import LongTensor

from Module import *

class LossMSE(Module):
    def forward(self, x_in, target):
        self.last_input = x_in
        self.last_target = target
        return (x_in.sub(target)).pow(2).mean()
    def backward(self):
        return 2*(self.last_input-self.last_target)/(self.last_target.size()[0]*self.last_target.size()[1])