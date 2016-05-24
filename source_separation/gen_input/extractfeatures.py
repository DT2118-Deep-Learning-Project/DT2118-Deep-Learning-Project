# -*- coding: utf-8 -*-
"""
Created on Mon May 23 22:47:13 2016

@author: flac
"""

import numpy as np
import tools
import sys
from pysndfile import sndio
import filefeatures as ff

import glob
import os
from os import listdir
from os.path import isfile, join


listFolder = ['tidigits_clean','tidigits_noise','tidigits_noisy']
prefix = '../../data/'
target= 'features'
slash = '/'
percent = 0.1

def extract(SoundPath):
    """
    Read a wav (nist format file included) and create the fft and mel on each window
    input
    SoundPath: wav file path
    return
    fft  and mel
    """
    sndobj = sndio.read(SoundPath)
    samples = np.array(sndobj[0])*np.iinfo(np.int16).max
    fft = tools.fftCoef(samples)
    return fft, tools.logMelSpectrum(fft, sndobj[1]) 

def extractfolder(folder, settype):
    """
    Go through a folder and extract the features for all wav files
    settype allow to choose between the train and test set
    """
    path = prefix + folder + slash + settype + slash + "man" + slash
    print path
    allfft = []
    allmel = []
    # Get all the wav files recursively (used glob)
    listwavfiles = [y for x in os.walk(path) for y in glob.glob(os.path.join(x[0], '*.wav'))]
    i=0
    total = int(len(listwavfiles)*percent)
    for filepath in listwavfiles:
        i=i+1
        filename = os.path.basename(filepath)
        filename = filename[0:-4] # Remove .wav
        print(str(i)+"/"+str(total)+ " samples")
        fft, mel = extract(filepath)
        allfft = {"filename": filename,
                     "fft": fft}
        allmel = {"filename": filename,
                     "fft": mel}
        ff.save_feature(prefix + target + slash + folder + slash + settype + slash + filename, allfft, allmel)
        if i > total:
            break

def createdir(directory):
    if not os.path.exists(directory):
            os.makedirs(directory)

def initfolder(directory, d1, d2):
    createdir(directory+d1)
    createdir(directory+d2)

"""
Create all the features
"""
directory = prefix + target
createdir(directory)
for folder in listFolder:
    directory = directory + slash + folder + slash
    createdir(directory)
    initfolder(directory, 'train', 'test') 
    #initfolder(directory+'/train/', 'man', 'woman') 
    #initfolder(directory+'/test/', 'man', 'woman') 
    extractfolder(folder, 'train')
    extractfolder(folder, 'test')
    directory = prefix + target
