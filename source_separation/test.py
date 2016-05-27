# -*- coding: utf-8 -*-
#!/usr/bin/python
# source_separation.py
# 
import numpy as np
import neural_network.RNN
import neural_network.loss_function
from preprocess import dataIO

setsize = 4 # Number of tidigits to take in the set, choose 0 to take all of them
prefix = "../data/"
hidden_layer = 2
nodes = 150
activation='relu'

def train_srs():
    # Load train set    
    print 'Loading files'
    X, Y = dataIO.train_set(setsize)

    # Create net
    print 'Building RNN'
    rnn = neural_network.RNN.RNN(input_size, hidden_layer, nodes, X, 2,
            loss=neural_network.loss_function.source_separation_loss_function, activation=activation)

    # Train net
    print 'Training'
    rnn.fit(X, Y, nb_epoch=2)

    # Save net
    print 'Saving'
    rnn.save()
    return rnn


def test_srs(rnn):
    # Load test set
    print 'Loading files'
    mergeFiles = True
    X, Y = dataIO.test_set()



    # Load net

    # Test net
    score = rnn.evaluate(X, Y)

    return score

