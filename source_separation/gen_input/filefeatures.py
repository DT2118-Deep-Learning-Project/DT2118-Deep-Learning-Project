#!/usr/bin/python
# inputFile.py
# Library for writing and reading the inputs for doing the source separation.
# For each set, there are two npz files: 
# 1. With the FFT (512)
# 2. With the Mel
# 
# Eric Masseran

import numpy as np
import os
import glob

def save_feature(folder, allfft):
    """ 
    Save the features on the disk.
    folder: place to save them
    allfft: fft features to save
    allmel: mel features to save
    """
    np.savez(folder + '_fft.npz', allfft=allfft)

def load_feature(prefix, settype, filename):
    """
    Load the features from the disk.
    input
    prefix: place where the features are saved
    settype: test or train
    filename: utterances
    return
    allfft: fft features
    allmel: mel features
    """
    folder = prefix + settype + slash + filename
    data = np.load(folder + '_fft.npz')
    allfft = data['allfft']
    data = np.load(folder + '_mel.npz')
    allmel = data['allmel']

    return allfft, allmel

def load_set_features(path, dtype):
    """
    Load all the files npz of the given path.
    The dtype allow to choose between fft (0)  or mel (1)
    """
    if dtype:
        filter = "mel"
    else:
        filter = "fft"
    files = [y for x in os.walk(path) for y in glob.glob(os.path.join(x[0], '*_' + filter + '.npz'))]
    set = []
    for f in files:
        data = np.load(f)
        if dtype:
            set.append(data['allmel'])
        else:
            set.append(data['allfft'])

    return set

def extract_fft(set_full, concatenate):
    """
    input
    List of numpy scalar dict with the fft and filename
    If concatenate (1) then all the fft are concatenate in one matrix else separated for each sample
    return
    Numpy array with only fft
    """
    set_fft = []
    for c in set_full:
        set_fft.append(c.item()['fft'])

    if concatenate:
        tmp = []
        for s in set_fft:
            for fft in s:
                tmp.append(fft)
        set_fft = tmp

    return np.array(set_fft)

