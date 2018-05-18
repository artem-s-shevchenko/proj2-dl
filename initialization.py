from torch import FloatTensor
import math

def scaled_init(w, b, init_type, gain=1):
    std = 1. / math.sqrt(w.size(1))
    if init_type == "normal":
        w.normal_(0, gain*std)
        b.normal_(0, gain*std)
    elif init_type == "uniform":
        temp = math.sqrt(3.0)*std*gain
        w.uniform_(-temp, temp)
        b.uniform_(-temp, temp)
        
def xavier_init(w, b, init_type, gain=1):
    fan_in = w.size(1)
    fan_out = w.size(0)
    std = math.sqrt(2.0 / (fan_in + fan_out))
    if init_type == "normal":
        w.normal_(0, gain*std)
        b.normal_(0, gain*std)
    elif init_type == "uniform":
        temp = math.sqrt(3.0)*std*gain
        w.uniform_(-temp, temp)
        b.uniform_(-temp, temp)

def calculate_gain(type):
    if type == 'tanh':
        return 5.0 / 3
    elif type == 'relu':
        return math.sqrt(2.0)
