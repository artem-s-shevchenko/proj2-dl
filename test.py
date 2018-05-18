from Linear import *
from ReLU import *
from LossMSE import *
from LossCrossEntropy import *
from Tanh import *
from Sequential import *
from SGD import *

from torch import FloatTensor
from torch import LongTensor 
import math

def convert_to_one_hot_labels(target):
    """
    Convert target tensor to one hot vector representation

    :param target: target tensor
    :return: one hot vector representation of target

    """
    tmp = FloatTensor(target.size(0), target.max() + 1).fill_(-1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def generate_disc_set(nb):
    """
    Generate the data

    :param nb: number of points in generated data
    :return: tensor with data and tensor with targets
    """
    uniform_input = FloatTensor(nb, 2).uniform_(0, 1)
    target = uniform_input.sub(0.5).pow(2).sum(1).sub(1 /(2*math.pi)).sign().mul(-1).add(1).div(2).long()
    return uniform_input, target

def train_model(model, train_input, train_target):
    """
    Train the model with specified data

    :param model: model to train
    :param train_input: tensor with data to use during training
    :param train_target: tensor with targets to use during training

    """
    #comment the first criterion and uncomment the second criterion to use CrossEntropy loss
    criterion = LossMSE()
    #criterion = LossCrossEntropy()
    nb_epochs = 250
    lr = 1e-1
    optim = SGD(model, lr)

    for e in range(0, nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
            dl_din = criterion.backward()
            model.backward(dl_din)
            optim.step()

def compute_nb_errors(model, data_input, data_target):
    """
    Test the model with specified data and compute number of errors
    
    :param model: model to test
    :param data_input: tensor with data to use during testing
    :param data_target: tensor with targets to use during testing
    :return: number of errors during testing

    """
    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model.forward(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.max(1)
        for k in range(0, mini_batch_size):
            #in case of using CrossEntropy comment the next line and uncomment the second line
            if data_target[b + k, predicted_classes[k]] < 0:
            #if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

#data generation
train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)
#in case of using CrossEntropy loss comment the next two lines
train_target = convert_to_one_hot_labels(train_target)
test_target = convert_to_one_hot_labels(test_target)
mini_batch_size = 100

#data noramlization
mean, std = train_input.mean(0), train_input.std(0)

train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

#create model
model = Sequential(
        Linear(2, 25), 
        Tanh(),
        Linear(25, 25), 
        Tanh(),
        Linear(25, 25), 
        Tanh(),
        Linear(25, 2)
)

#train network
train_model(model, train_input, train_target)

#compute and print the error
print('train_error {:.02f}% test_error {:.02f}%'.format(
            compute_nb_errors(model, train_input, train_target) / train_input.size(0) * 100,
            compute_nb_errors(model, test_input, test_target) / test_input.size(0) * 100
        )
        )

