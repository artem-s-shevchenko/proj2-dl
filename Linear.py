from torch import FloatTensor
from torch import LongTensor 

from Module import *

class Linear (Module):
    def __init__(self, in_features, out_features):
        """
        Class of Linear module

        :param in_features: size of each input sample
        :param out_features: size of each output sample
        """
        super(Module, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = FloatTensor(out_features, in_features).normal_()
        self.b = FloatTensor(1, out_features).normal_()
        self.dl_dw = FloatTensor(self.w.size())
        self.dl_db = FloatTensor(self.b.size())  
    def forward(self, x_in):
        """
        Forward pass of module

        :param x_in: input tensor
        :return: result of forward pass
        """
        x_out = x_in.mm(self.w.t())+self.b
        self.x_in = x_in
        return x_out
    def backward(self, dl_dx_out):
        """
        Backward pass of module

        :param dl_dx_out: tensor containing the gradient of the loss with respect to the module’s output
        :return: tensor containing the gradient of the loss wrt the module’s input
        """
        dl_dw = dl_dx_out.t().mm(self.x_in)
        dl_db = dl_dx_out.sum(0).view(1,-1)
        dl_dx_in = dl_dx_out.mm(self.w)
        self.dl_dw = dl_dw
        self.dl_db= dl_db
        return dl_dx_in
    def param(self):
        """
        Parameters of module

        :return: return a list of 2 pairs, each composed of a parameter tensor and a gradient tensor of same size
        """
        return [(self.w, self.dl_dw), (self.b, self.dl_db)]

