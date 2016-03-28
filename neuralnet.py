__author__ = 'Hussam_Qassim'

import numpy as np

import theano
import lasagne
import theano.tensor as T


# This path should be reset when changing computer
root = "/home/hualkassam/Desktop/Dataset"

pixelNo = 128
num_epochs = 100

input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
relu = lasagne.nonlinearities.rectify


def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_indexes in range(0, len(inputs)-batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_indexes:start_indexes + batchsize]
        else:
            excerpt = slice(start_indexes, start_indexes + batchsize)
        yield inputs[excerpt], targets[excerpt]

		
def iterate_batches(inputs, batchsize):
    for start_indexes in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_indexes, start_indexes + batchsize)
        yield inputs[excerpt]
		
		
def build_model():

    inp = lasagne.layers.InputLayer(shape=(None, 1, pixelNo, pixelNo), input_var=input_var)

    conv = lasagne.layers.Conv2DLayer(inp,  num_filters=8, filter_size=(3, 3), nonlinearity=relu)
    conv = lasagne.layers.MaxPool2DLayer(conv, pool_size=(2, 2))
	
    conv = lasagne.layers.Conv2DLayer(conv, num_filters=8, filter_size=(3, 3), nonlinearity=relu)
    conv = lasagne.layers.MaxPool2DLayer(conv, pool_size=(2, 2))
	
    conv = lasagne.layers.Conv2DLayer(conv, num_filters=8, filter_size=(3, 3), nonlinearity=relu)
    conv = lasagne.layers.MaxPool2DLayer(conv, pool_size=(2, 2))

    hid1 = lasagne.layers.DenseLayer(conv, num_units=16, nonlinearity=relu)
    hid1 = lasagne.layers.DropoutLayer(hid1, p=0.5)
    hid2 = lasagne.layers.DenseLayer(hid1, num_units=16, nonlinearity=relu)
    hid2 = lasagne.layers.DropoutLayer(hid2, p=0.5)

	
    out = lasagne.layers.DenseLayer(hid2, num_units=3, nonlinearity=lasagne.nonlinearities.softmax)

    return out

