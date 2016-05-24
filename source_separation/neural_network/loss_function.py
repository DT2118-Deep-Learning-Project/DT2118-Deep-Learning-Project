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

gamma=0.5
sizesource=512

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
    y1_true = y_true[0:sizesource]
    y2_true = y_true[sizesource: ]
    y1_pred = y_pred[0:sizesource]
    y2_pred = y_pred[sizesource: ]
    # Compute mean squared error
    p1 = mean_squared_error(y1_pred, y1_true)
    p2 = mean_squared_error(y1_pred, y2_true)
    p3 = mean_squared_error(y2_pred, y2_true)
    p4 = mean_squared_error(y1_pred, y1_true)

    return (p1+p3) - gamma*(p2+p4)
