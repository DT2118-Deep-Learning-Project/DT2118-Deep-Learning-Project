# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from output_layer import Output_Layer # Custom output layer 
import numpy as np
from keras.utils.np_utils import to_categorical
from mask_data_callback import Mask_Data_Callback 

# generate dummy data
data = np.random.random((1000, 16))
labels = data + 0.05 * np.random.random((1000, 16))

# train the model, iterating on the data in batches
# of 32 samples

model = Sequential()
model.add(Dense(16, input_dim=16, activation='softmax'))
model.add(Dense(32, activation='softmax'))
model.add(Output_Layer(16, data))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, nb_epoch=10, batch_size=1, callbacks=[Mask_Data_Callback()])