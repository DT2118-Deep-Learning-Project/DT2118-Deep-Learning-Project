# Need to download python's library h5py
# for the save/load models
# http://docs.h5py.org/en/latest/build.html


import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import SimpleRNN, Dense
from keras.utils.visualize_util import plot


def genRNN(insize, hidden_layer, outsize, timesteps):
    model = Sequential()
    model.add(SimpleRNN(outsize,input_shape=(timesteps, insize),return_sequences=True, activation='relu'))
    for i in range(hidden_layer):
        model.add(SimpleRNN(outsize, return_sequences=True, activation = 'relu'))
    model.add(SimpleRNN(outsize, return_sequences=False, activation = 'relu'))
    model.add(Dense(outsize))
    return model
    
def printStructure(model, name='model'):
    plot(model, name + '.png', show_shapes = True)
    
def save(model, name, overwrite = False):
    json_string = model.to_json()
    open(name + '.json', 'w').write(json_string)
    model.save_weights(name + 'weights.h5', overwrite)
    
def load(name):
    m = model_from_json(open(name + '.json').read())
    m.load_weights(name + '_weights.h5')
    return m
    
def genDumbData():
    X = np.ones((100,2))
    Y = np.ones(100)
    return X, Y
    
def load_data(data, n_prev = 3):  
    docX = []
    for i in range(len(data)-n_prev):
        docX.append(data[i:i+n_prev,:])
    return np.array(docX)



