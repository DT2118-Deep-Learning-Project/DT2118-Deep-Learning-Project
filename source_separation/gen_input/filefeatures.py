#!/usr/bin/python
# inputFile.py
# Library for writing and reading the inputs for doing the source separation.
# For each set, there are two npz files: 
# 1. With the FFT (512)
# 2. With the Mel
# 
# Eric Masseran

import numpy as np

def save_feature(folder, allfft, allmel):
    """ 
    Save the features on the disk.
    folder: place to save them
    allfft: fft features to save
    allmel: mel features to save
    """
    np.savez(folder + '_fft.npz', allfft=allfft)
    np.savez(folder + '_mel.npz', allmel=allmel)

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
