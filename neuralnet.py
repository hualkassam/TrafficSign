__author__ = 'Hussam_Qassim'

import numpy as np # Import numpy for array creation

import theano # Import the Theano to allow us to work with multi-dimensional arrays efficiently
import lasagne # Import the lasagne
import theano.tensor as T # Set Theano tensor to T


# This path should be reset when changing computer
root = "/home/hualkassam/Desktop/Dataset"

pixelNo = 128 # Number of the input image pixels
num_epochs = 30 # Number of epochs

input_var = T.tensor4('inputs')
target_var = T.ivector('targets')
Tanh = lasagne.nonlinearities.tanh # Non-linear activation functions for artificial neurons


def iterate_minibatches(inputs, targets, batchsize, shuffle=True): # Batch Learning mode
    assert len(inputs) == len(targets) # Python's assert statement helps you find bugs more quickly and with less pain
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_indexes in range(0, len(inputs)-batchsize + 1, batchsize): # This is a versatile function to create lists containing arithmetic progressions
        if shuffle:
            excerpt = indices[start_indexes:start_indexes + batchsize]
        else:
            excerpt = slice(start_indexes, start_indexes + batchsize) # Return a slice object representing the set of indices specified
        yield inputs[excerpt], targets[excerpt]


def iterate_batches(inputs, batchsize):# Batch learning mode_Iteration
    for start_indexes in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_indexes, start_indexes + batchsize)
        yield inputs[excerpt]


def build_model(): # Build the neural network model

    inp = lasagne.layers.InputLayer(shape=(None, 1, pixelNo, pixelNo), input_var=input_var) # This layer holds a symbolic variable that represents a network input

    conv = lasagne.layers.Conv2DLayer(inp,  num_filters=60, filter_size=(3, 3), nonlinearity=Tanh) # 2D convolutional layer
    conv = lasagne.layers.MaxPool2DLayer(conv, pool_size=(6, 6)) # 2D max-pooling layer
	
    conv = lasagne.layers.Conv2DLayer(conv, num_filters=60, filter_size=(3, 3), nonlinearity=Tanh)
    conv = lasagne.layers.MaxPool2DLayer(conv, pool_size=(6, 6))


    hid1 = lasagne.layers.DenseLayer(conv, num_units=120, nonlinearity=Tanh) # A fully connected layer
    hid1 = lasagne.layers.DropoutLayer(hid1, p=0.5) # Noise layers. The dropout layer is a regularizer
    hid2 = lasagne.layers.DenseLayer(hid1, num_units=60, nonlinearity=Tanh)
    hid2 = lasagne.layers.DropoutLayer(hid2, p=0.5)

	
    out = lasagne.layers.DenseLayer(hid2, num_units=10, nonlinearity=lasagne.nonlinearities.softmax) # Output layer. Softmax activation function. 10 classes

    return out

