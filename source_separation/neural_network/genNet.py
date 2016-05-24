# -*- coding: utf-8 -*-
"""
Created on Tue May 24 14:05:14 2016

@author: flac
"""

from keras.models import Sequential 
from keras.layers import Dense, SimpleRNN , Activation
from keras.utils.visualize_util import plot



def genRNN(insize, hidden_layer, outsize, timesteps):
    model = Sequential()
    model.add(SimpleRNN(outsize,input_shape=(timesteps, insize),return_sequences=True, activation='relu'))
    for i in range(hidden_layer):
        model.add(SimpleRNN(outsize,return_sequences=True, activation = 'relu'))
    #Ajouter la dernière couche de Marc
    return model

m = genRNN(40,3,40,2)
l = m.layers[0]
plot(m, 'model.png',show_shapes = True)