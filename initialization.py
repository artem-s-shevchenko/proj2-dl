from torch import FloatTensor
import math

def scaled_init(w, b, init_type, gain=1):
    """
    Initializes weights and bias with variance depending on the number of neurons in the previous layer 

    :param w: weights tensor
    :param b: bias tensor
    :param init_type: "normal" or "uniform" initialization
    :param gain: optional scaling factor
    """
    std = 1. / math.sqrt(w.size(1))
    if init_type == "normal":
        w.normal_(0, gain*std)
        b.normal_(0, gain*std)
    elif init_type == "uniform":
        temp = math.sqrt(3.0)*std*gain
        w.uniform_(-temp, temp)
        b.uniform_(-temp, temp)
        
def xavier_init(w, b, init_type, gain=1):
    """
    Initializes weights and bias according to the Xavier initialization

    :param w: weights tensor
    :param b: bias tensor
    :param init_type: "normal" or "uniform" initialization
    :param gain: optional scaling factor
    """
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
    """
    Returns the gain value for the given nonlinearity function

    :param type: name of nonlinearity function
    """
    if type == 'tanh':
        return 5.0 / 3
    elif type == 'relu':
        return math.sqrt(2.0)
