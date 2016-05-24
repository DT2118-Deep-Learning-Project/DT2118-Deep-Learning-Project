# -*- coding: utf-8 -*-
from keras.callbacks import Callback

class Mask_Data_Callback(Callback):
    '''
        This callback send the current batch to the last layer
        so that masking can be done on the original data
    '''
    idx = 0

    def __init__(self, idx=0):
        Mask_Data_Callback.idx = idx

    def on_batch_begin(self, batch, logs={}):
        Mask_Data_Callback.idx += 1

    def on_batch_end(self, batch, logs={}):
        print " Batch ends / idx:", Mask_Data_Callback.idx

