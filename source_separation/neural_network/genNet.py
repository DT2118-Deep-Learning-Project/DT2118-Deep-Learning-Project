# Need to download python's library h5py
# for the save/load models
# http://docs.h5py.org/en/latest/build.html


import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import SimpleRNN, Dense
from keras.utils.visualize_util import plot
from loss_function import *

def rnn(insize, outsize, hidden_layer, timesteps):
    model = Sequential()
    model.add(SimpleRNN(outsize,input_shape=(timesteps, insize),return_sequences=True, activation='relu'))
    for i in range(hidden_layer):
        model.add(SimpleRNN(150, return_sequences=True, activation = 'relu'))
    model.add(SimpleRNN(outsize, return_sequences=False, activation = 'relu'))
    model.add(Dense(outsize))
    return model
    
def dnn(insize = 512, ousize = 1024, hidden_size = [1024, 1024, 1024]):
    model = Sequential()
    model.add(Dense(hidden_size[0], input_dim=insize, init='uniform', activation = 'sigmoid'))
    for i in range(1,len(hidden_size)):
        model.add(Dense(hidden_size[i], init='uniform', activation = 'sigmoid'))
    model.add(Dense(ousize, input_dim=insize, init='uniform', activation = 'sigmoid'))
    return model
    
def printStructure(model, name='model'):
    plot(model, name + '.png', show_shapes = True)
    
def save_model(model, name, overwrite = False):
    json_string = model.to_json()
    open(name + '.json', 'w').write(json_string)
    model.save_weights(name + 'weights.h5', overwrite)
    
def load_model(name):
    m = model_from_json(open(name + '.json').read())
    m.load_weights(name + '_weights.h5')
    return m
    
def prepare_data(data, n_prev = 3):  
    docX = []
    for i in range(len(data)-n_prev):
        docX.append(data[i:i+n_prev,:])
    return np.array(docX)
     

#insize = 1024
#hidden_layer = 3
#outsize = 1024
#timesteps = 2
#m = genRNN(insize, hidden_layer, outsize, timesteps)
##printStructure(m)
#
#X,Y = genDumbData()
#X = load_data(X,3)
#Y = Y[1:98]
#Y = np.array([Y,Y])
#Y = Y.T
#m.compile(optimizer='rmsprop',loss=source_separation_loss_function)
#m.fit(X, Y, nb_epoch = 3,batch_size=1)
