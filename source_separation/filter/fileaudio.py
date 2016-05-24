#!/usr/bin/python
"""
Functions for dealing with wav files
"""

import os
import glob
import numpy as np 
import scipy.io.wavfile as sw

def load_set_wav(path):
    """
    Load all the wav files in the path directory.
    return:
    Numpy array with all the wav files
    """
    files = [y for x in os.walk(path) for y in glob.glob(os.path.join(x[0], '*.wav'))]
    set = []
    for f in files:
        fs, data = sw.read(f)
        set.append(data)

    return np.array(set)

def save_set_wav(path, wav_filename, wav_data):
    """
    Save the wav files on the disk.
    input:
    path: main path
    wav_filename: np array with the file name inside
    wav_data: np array of wav files
    """
    for i in range(wav_filename.size):
        sw.write(path+wav_filename[i], 20000, wav_data[0])
    
