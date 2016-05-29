# -*- coding: utf-8 -*-
import os
import glob
import shutil
from wav_fft import readFFT, writeFFT, writeWAV
import numpy as np
import random

def createdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extractAllFFTfromWAV(src_path, destination, percent=0.05):
    """
        Read all the wav files in src_path
        Convert it to spectrums
        Writes the STFT into destination, splitting data into train/test
    """
    
    sound_type = ['clean', 'noise']
    data_type  = ['train', 'test']

    # Set up directories, removing first if data already exists
    if(os.path.exists(destination)):
        shutil.rmtree(destination)
    createdir(destination)

    # Going through all the subfolders (clean/train, clean/test, noise/train...)
    for s_type in sound_type:
        createdir(destination + '/tidigits_' + s_type)
        for d_type in data_type:
            createdir(destination + '/tidigits_' + s_type + '/' + d_type)

            # Get all the wav files recursively (used glob)
            path = src_path + '/tidigits_' + s_type + '/' + d_type
            listwavfiles = [y for x in os.walk(path) for y in glob.glob(os.path.join(x[0], '*.wav'))]
            i = 0
            total = int(len(listwavfiles)*percent)

            for filepath in listwavfiles:
                print s_type, "-", d_type, i, "/", total
                # Don't read too much files
                if i > total:
                    break
                i += 1
                # Read file, convert it
                filename = os.path.basename(filepath)
                stft_data = readFFT(filepath)
                # Save STFT to e.g. wav/tidigits_clean/train/
                filename_npz = filename[:-4]
                writeFFT(destination + '/tidigits_' + s_type + '/' + d_type, filename_npz, stft_data)

def loadFFTfromFiles(src_path):
    """
        Load all the npz files, assuming src_path contains
        clean & noise directories & train + test in each of them

        return 2 lists 'clean' and 'noise' for which each element is a numpy array
        representing the FFT from one file
    """
    sound_type = ['clean', 'noise']
    data_type  = ['train', 'test']

    # Going through all the subfolders (clean/train, clean/test, noise/train...)
    STFTs = {}
    for s_type in sound_type:
        s_type_stft = {}
        for d_type in data_type:
            d_type_stft = []

            path = src_path + '/tidigits_' + s_type + '/' + d_type
            listnpzfiles = [y for x in os.walk(path) for y in glob.glob(os.path.join(x[0], '*.npz'))]
            for filename in listnpzfiles:
                f = np.load(filename)
                d_type_stft.append(f['fft'])

            s_type_stft[d_type] = d_type_stft

        STFTs[s_type] = s_type_stft

    return STFTs['clean'], STFTs['noise']

def saveFFTlistToWAVs(stft_data, destination):
    """
        Convert & save each STFTs into a wav files in the destination directory
        Names are just the index in the list + '.wav'

        :param stft_data: list of numpy arrays of size (N_frames, STFT_length)
    """
    createdir(destination)
    for i, stft in enumerate(stft_data):
        writeWAV(destination + '/' + str(i) + '.wav', stft_data)

def train_set(setsize=0):
    '''
        Return train set, cropped to setsize, or the whole set if setsize == 0 
    '''
    clean, noise = loadFFTfromFiles('../data/features')
    clean, noise = clean['train'], noise['train']

    if setsize == 0:
        setsize = len(clean)

    cl2 = clean[:]
    random.shuffle(clean)

    Y = []
    X = []
    stft_len = noise[0].shape[1]
    for z in zip(clean[:setsize], cl2[:setsize]):
        max_size = min([z[0].shape[0], z[1].shape[0]])
        sound = np.concatenate((z[0][:max_size], z[1][:max_size]), axis=1)
        for frame in sound:
            Y.append(frame)
            X.append(frame[:stft_len] + frame[stft_len:])
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

def test_set(setsize=0):
    '''
        Return the test_set as a list of noisy & clean FFT
        The return values X, Y are 2 lists. Each element contains the
        STFTs for one file.

        So note that the STFTs are not put into one big 2D array 
        as they are for the train set.
    '''
    clean, noise = loadFFTfromFiles('../data/features')
    clean, noise = clean['test'], noise['test']

    if setsize == 0:
        setsize = len(clean)

    Y = []
    X = []
    stft_len = noise[0].shape[1]
    for z in zip(clean[:setsize], noise[:setsize]):
        Y.append(np.concatenate((z[0], z[1]), axis=1))
        X.append(z[0] + z[1])
    X = np.array(X)
    Y = np.array(Y)

    return X, Y
