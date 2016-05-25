# -*- coding: utf-8 -*-
from keras.callbacks import Callback

class Mask_Data_Callback(Callback):
    '''
        This callback send the current batch to the last layer
        so that masking can be done on the original data
    '''
    idx = 0
    batch_size = 0

    def __init__(self, batch_size, idx=0):
        Mask_Data_Callback.idx = idx
        Mask_Data_Callback.batch_size = batch_size

    def on_batch_begin(self, batch, logs={}):
        Mask_Data_Callback.idx += 1

    def on_batch_end(self, batch, logs={}):
        if Mask_Data_Callback.idx == Mask_Data_Callback.batch_size:
            Mask_Data_Callback.idx = 0

