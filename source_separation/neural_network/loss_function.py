#!/usr/bin/python
"""
Masseran Eric
Loss function from the paper:
    DEEP LEARNING FOR MONAURAL SPEECH SEPARATION
    Po-Sen Huang , Minje Kim , Mark Hasegawa-Johnson , Paris Smaragdis 
    p3 - 3.3
"""
import numpy as np
from keras import backend as K
import theano.tensor as tt

gamma=0.05
sizesource=1

def split_half(x, axis=0):
    size1 = x.shape[axis] / 2
    size2 = x.shape[axis] - size1
    return tt.split(x, [size1, size2], 2, axis=axis)

def mean_squared_error(y_true, y_pred):
    """
    Compute mean square
    """
    return K.mean(K.square(y_pred - y_true), axis=-1)
    
def source_separation_loss_function(y_true, y_pred):
    """
    Need gloabal size for splitting data
    Need global coef gamma
    """
    y1_true, y2_true = split_half(y_true, 1)
    y1_pred, y2_pred = split_half(y_pred, 1)
    # Compute mean squared error
    p1 = mean_squared_error(y1_pred, y1_true)
    p2 = mean_squared_error(y1_pred, y2_true)
    p3 = mean_squared_error(y2_pred, y2_true)
    p4 = mean_squared_error(y2_pred, y1_true)
    return (p1+p3) - gamma*(p2+p4)

