# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from output_layer import Output_Layer # Custom output layer 
import numpy as np
from keras.utils.np_utils import to_categorical
from mask_data_callback import Mask_Data_Callback 

INPUT_SIZE  = 1024
OUTPUT_SIZE = 2 * INPUT_SIZE
SET_SIZE    = 1000

# Generate dummy data
clean = np.random.random((SET_SIZE, INPUT_SIZE))
noise = 0.05 * np.random.random((SET_SIZE, INPUT_SIZE))

noisy  = clean + noise
target = np.append(clean, noise, axis=1)

# Building the classifier
model = Sequential()
model.add(Dense(INPUT_SIZE, input_dim=INPUT_SIZE, activation='softmax'))
model.add(Dense(OUTPUT_SIZE, activation='softmax'))
model.add(Output_Layer(INPUT_SIZE, noisy))

model.compile(optimizer='sgd', loss='mse')

print target.shape[0]
mask_data = Mask_Data_Callback(target.shape[0])
model.fit(noisy, target, nb_epoch=10, batch_size=1, callbacks=[mask_data])
