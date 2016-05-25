# -*- coding: utf-8 -*-
"""
Created on Wed May 25 18:40:45 2016

@author: flac
"""

import scipy.fftpack
import numpy as np
from filefeature.py

data = np.load('1b_fft.npz')


def fromffttowav(data):
    res = scipy.fftpack.ifft(data)
    
    return res
    