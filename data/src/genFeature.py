# -*- coding: utf-8 -*-
"""
Created on Mon May 23 22:47:13 2016

@author: flac
"""

import numpy as np
import tools
from pysndfile import sndio

from os import listdir
from os.path import isfile, join


def extract(SoundPath):
    sndobj = sndio.read(SoundPath)
    samples = np.array(sndobj[0])*np.iinfo(np.int16).max
    fft = tools.fftCoef(samples)
    return fft, tools.logMelSpectrum(fft, sndobj[1]) 

prefix = '../'
listFolder = ['tidigits_clean','tidigits_noise','tidigits_noisy']
slash = '/'


for folder in listFolder:
    path = prefix + folder + slash
    allfft = []
    allmel = []
    listSounds = [f for f in listdir(path) if isfile(join(path, f))]
    for filename in listSounds:
        print path + filename
        fft, mel = extract(path + filename)
        allfft.append({"filename": filename,
                     "fft": fft})
        allmel.append({"filename": filename,
                     "fft": mel})
    np.savez(folder + '_fft.npz', allfft=allfft)
    np.savez(folder + '_mel.npz', allmel=allmel)

