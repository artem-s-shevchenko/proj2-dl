from torch import FloatTensor
from torch import LongTensor 

from Module import *

class Linear (Module):
    def __init__(self, in_features, out_features):
        super(Module, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = FloatTensor(in_features, out_features).normal_(0, epsilon)
        self.b = FloatTensor(in_features, 1).normal_(0, epsilon) #TODO maybe put some specific parameters for init, 
                                                            #try xavier
        self.dl_dw = FloatTensor(self.w.size())
        self.dl_db = FloatTensor(self.b.size())  
    def forward(self, x_in):
        x_out = x_in.mm(self.w.t())+self.b.t()
        self.x_in = x_in
        return x_out
    def zero_grad(self): 
        self.dl_dw.zero_()
        self.dl_db.zero_()
    def backward(self, dl_dx_out): 
        dl_dw = 1/self.dl_dw.shape[0]*dl_dx_out.t().mm(self.x_in)
        dl_db = 1/self.dl_db.shape[0]*dl_dx_out.sum(0).view(-1,1)
        dl_dx_in = dl_dx_out.mm(self.w)
        self.dl_dw = dl_dw
        self.dl_db= dl_db
        return dl_dx_in
    def param(self): 
        return [(self.w, self.dl_dw), (self.b, self.dl_db)]

