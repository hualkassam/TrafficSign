__author__ = 'Hussam_Qassim'

import numpy as np # http://docs.scipy.org/doc/numpy-1.10.1/user/basics.creation.html

import theano
import lasagne
import theano.tensor as T


# This path should be reset when changing computer
root = "/home/hualkassam/Desktop/Dataset"

pixelNo = 128
num_epochs = 30

input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
relu = lasagne.nonlinearities.tanh # http://lasagne.readthedocs.org/en/latest/modules/nonlinearities.html


def iterate_minibatches(inputs, targets, batchsize, shuffle=True): # Batch Learning
    assert len(inputs) == len(targets) # https://wiki.python.org/moin/UsingAssertionsEffectively
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_indexes in range(0, len(inputs)-batchsize + 1, batchsize): # https://docs.python.org/2/library/functions.html#range
        if shuffle:
            excerpt = indices[start_indexes:start_indexes + batchsize]
        else:
            excerpt = slice(start_indexes, start_indexes + batchsize) # https://docs.python.org/2/library/functions.html#slice
        yield inputs[excerpt], targets[excerpt] # https://wiki.python.org/moin/Generators


def iterate_batches(inputs, batchsize):
    for start_indexes in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_indexes, start_indexes + batchsize)
        yield inputs[excerpt]


def build_model(): # http://lasagne.readthedocs.org/en/latest/modules/layers.html

    inp = lasagne.layers.InputLayer(shape=(None, 1, pixelNo, pixelNo), input_var=input_var) # http://lasagne.readthedocs.org/en/latest/modules/layers/input.html

    conv = lasagne.layers.Conv2DLayer(inp,  num_filters=60, filter_size=(3, 3), nonlinearity=relu) # http://lasagne.readthedocs.org/en/latest/modules/layers/conv.html
    conv = lasagne.layers.MaxPool2DLayer(conv, pool_size=(6, 6)) # http://lasagne.readthedocs.org/en/latest/modules/layers/pool.html#lasagne.layers.MaxPool2DLayer
	
    conv = lasagne.layers.Conv2DLayer(conv, num_filters=60, filter_size=(3, 3), nonlinearity=relu)
    conv = lasagne.layers.MaxPool2DLayer(conv, pool_size=(6, 6))


    hid1 = lasagne.layers.DenseLayer(conv, num_units=120, nonlinearity=relu) # http://lasagne.readthedocs.org/en/latest/modules/layers/dense.html#lasagne.layers.DenseLayer
    hid1 = lasagne.layers.DropoutLayer(hid1, p=0.5) # http://lasagne.readthedocs.org/en/latest/modules/layers/noise.html#lasagne.layers.DropoutLayer
    hid2 = lasagne.layers.DenseLayer(hid1, num_units=120, nonlinearity=relu)
    hid2 = lasagne.layers.DropoutLayer(hid2, p=0.5)

	
    out = lasagne.layers.DenseLayer(hid2, num_units=10, nonlinearity=lasagne.nonlinearities.softmax) # http://lasagne.readthedocs.org/en/latest/modules/nonlinearities.html#lasagne.nonlinearities.softmax

    return out

