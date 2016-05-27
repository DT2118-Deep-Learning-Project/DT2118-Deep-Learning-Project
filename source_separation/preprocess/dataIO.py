# -*- coding: utf-8 -*-
import os
import glob
import shutil
from wav_fft import readFFT, writeFFT
import numpy as np

def createdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extractAllFFTfromWAV(src_path, destination, percent=0.05):
    """
        Read all the wav files in src_path
        Convert it to spectrums
        Writes the STFT into destination, splitting data into train/test
    """
    
    sound_type = ['clean', 'noise', 'noisy']
    data_type  = ['train', 'test']

    # Set up directories, removing first if data already exists
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
        clean, noise, noisy directories & train + test in each of them
        
        returns a dictionnary with keys clean / noise / noisy with elements
        another dictionnary with keys train / test, with elements a 3D array with the STFT

        E.g. one can write: loadFFTfromFiles('path')['clean']['train'][99], which correspond to the
        STFT of the clean/train tidigit from the 100th file.

        So loadFFTfromFiles('path')['clean']['train'][99] is a np array of size (N_windows, 512)
    """
    sound_type = ['clean', 'noise', 'noisy']
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

    return STFTs['clean'], STFTs['noise'], STFTs['noisy']

def saveAllFFTtoWAV(stft_data, destination):
    """
        Convert & save the STFTs into wav files in the destination directory
    """
    print "WIP...."
    createdir(destination)

def set(set_type, setsize=0):
    '''
        Return either train or test set according the the set_type parameter
    '''
    clean, noise, noisy = loadFFTfromFiles('../data/features')
    clean, noise, noisy = clean[set_type], noise[set_type], noisy[set_type]

    if setsize == 0:
        setsize = len(noisy)

    X = []
    # Let's build a big 2D array with all the frames
    for sound in noisy[:setsize]:
        for frame in sound:
            X.append(frame)
    X = np.array(X)

    Y = []
    frame_size = X.shape[1]
    for z in zip(clean[:setsize], noise[:setsize]):
        sound = np.dstack((z[0], z[1]))
        sound = sound.reshape((sound.shape[0], sound.shape[1] * 2))
        for frame in sound:
            Y.append(frame)
    Y = np.array(Y)

    return X, Y

def train_set(setsize=0):
    return set('train', setsize)

def test_set(setsize=0):
    return set('test', setsize)
