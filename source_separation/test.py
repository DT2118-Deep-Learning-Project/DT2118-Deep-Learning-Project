# -*- coding: utf-8 -*-
#!/usr/bin/python
# source_separation.py
# 
import numpy as np
import os
import neural_network.RNN as RNN
import neural_network.loss_function
from preprocess import dataIO

setsize = 10 # Number of tidigits to take in the set, choose 0 to take all of them
nb_epoch=10
prefix = os.environ['SPEECH_PROJ_DIR']+"/data/"
hidden_layer = 2
nodes = 150
activation='relu'

def train_srs():
    # Load train set    
    print 'Loading files'
    X, Y = dataIO.train_set(setsize)


    scale = (np.mean(Y) + np.mean(X)) / 2
    Y = Y / scale
    X = X / scale

    # Create net
    print 'Building RNN'
    rnn = RNN.RNN(512, hidden_layer, nodes, X, 2,
            loss=neural_network.loss_function.source_separation_loss_function, activation=activation)

    # Train net
    print 'Training'
    rnn.fit(X, Y, nb_epoch=nb_epoch)

    # Save net
    print 'Saving'
    rnn.save()
    return rnn

def test_srs(rnn):
    # Load test set
    print 'Loading files'
    X, Y = dataIO.test_set()

    # Load net

    # Test net
    score = rnn.evaluate(X, Y)

    return score

# Run a train if used as a script
if __name__ == '__main__':
    train_srs()
