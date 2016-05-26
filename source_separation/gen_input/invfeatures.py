# -*- coding: utf-8 -*-
"""
Created on Wed May 25 18:40:45 2016

@author: flac
"""

import scipy.fftpack
import numpy as np

def istft(X, fs, T, hop=0.010):
     x = scipy.zeros(T*fs)
     framesamp = X.shape[1]
     hopsamp = int(hop*fs)
     for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
         x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
     return x

def fromffttowav(FFTs):
    fs = 20000
    hop = 0.01
    T = FFTs.shape[0] * hop
    samples = istft(FFTs, 20000, T, hop)
    return samples
    
