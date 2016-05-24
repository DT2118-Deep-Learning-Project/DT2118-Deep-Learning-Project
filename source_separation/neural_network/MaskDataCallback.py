# -*- coding: utf-8 -*-

class MaskDataCallback(keras.callbacks.Callback):
    '''
        This callback send the current batch to the last layer
        so that masking can be done on the original data
    '''
    def on_batch_begin(self, batch, logs={}):
        self.model.layers.

