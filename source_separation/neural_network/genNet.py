# Need to download python's library h5py
# for the save/load models
# http://docs.h5py.org/en/latest/build.html


import numpy as np
import os
import glob
from keras.models import Sequential, model_from_json
from keras.layers import SimpleRNN, Dense
from keras.utils.visualize_util import plot
from loss_function import *

def genRNN(insize, outsize, hidden_layer, timesteps):
    model = Sequential()
    model.add(SimpleRNN(outsize,input_shape=(timesteps, insize),return_sequences=True, activation='relu'))
    for i in range(hidden_layer):
        model.add(SimpleRNN(150, return_sequences=True, activation = 'relu'))
    model.add(SimpleRNN(outsize, return_sequences=False, activation = 'relu'))
    model.add(Dense(outsize))
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
    
def genDumbData():
    X = np.ones((100,2))
    Y = np.ones(100)
    return X, Y
    
def prepare_data(data, n_prev = 3):  
    docX = []
    for i in range(len(data)-n_prev):
        docX.append(data[i:i+n_prev,:])
    return np.array(docX)

def load_data():
    opath = '../../data/features/tidigits_'
    listfolder = ['clean', 'noise', 'noisy']
    allfft = []
    allmel = []
    for folder in listfolder:
        path = opath + folder + '/' + 'train/'
        print path
        listwavfiles = [y for x in os.walk(path) for y in glob.glob(os.path.join(x[0], '*.npz'))]
#        listwavfiles = os.listdir(path)
        print len(listwavfiles)
        for files in listwavfiles:
            temp = np.load(files)
            if (listwavfiles[-7:-4] == 'fft'):
                allfft.append(temp)
            else:
                allmel.append(temp)   
    return allfft, allmel
     
     

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
X,Y = load_data()