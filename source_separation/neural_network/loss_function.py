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

gamma=0.5
#sizesource=512

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
    sizesource = y_true.size.eval()/2
    y1_true, y2_true = tt.split(y_true, [sizesource, sizesource], 2)
    y1_pred, y2_pred = tt.split(y_pred, [sizesource, sizesource], 2)
    # Compute mean squared error
    p1 = mean_squared_error(y1_pred, y1_true)
    p2 = mean_squared_error(y1_pred, y2_true)
    p3 = mean_squared_error(y2_pred, y2_true)
    p4 = mean_squared_error(y1_pred, y1_true)
    return (p1+p3) - gamma*(p2+p4)

