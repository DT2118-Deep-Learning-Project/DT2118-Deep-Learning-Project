# -*- coding: utf-8 -*-
"""
    This file can load WAV files, convert samples to STFT / FFT,
    and writes features data to be later used again
"""
import scipy.io.wavfile
import numpy as np

FRAMESZ = 0.0256 # window length in ms
HOP     = 0.010 # window interval in ms
FS      = 20000 # sample rate

def stft(x, fs, framesz=FRAMESZ, hop=HOP):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hanning(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp])
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X
 
def istft(X, fs, T, hop=HOP):
    x = scipy.zeros(T*fs)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    return np.int16(x)

def readFFT(SoundPath):
    """
    Read a wav (nist format file included) and create the fft and mel on each window
    input
    SoundPath: wav file path
    return
    fft  and mel
    """
    fs, samples = scipy.io.wavfile.read(SoundPath)
    ft = stft(samples, fs, FRAMESZ, HOP)
    return np.int16(ft)

def writeWAV(path, stft_data):
    """
        Convert the stft into wav samples and write it to the specified path
    """
    T = stft_data.shape[0] * HOP
    print T
    samples = istft(stft_data, FS, T, HOP)
    print samples.shape
    scipy.io.wavfile.write(path, FS, samples)

def writeFFT(path, filename, stft_data):
    """
        Save STFTs into a .npz file
    """
    np.savez(path + '/' + filename + '_fft.npz', filename=filename, fft=stft_data)

def apply_filter(mix, mask_1, mask_2):
    """
    mix: np.array sound with source 1 and 2 mixed
    mask_1: np.array mask for the source 1
    mask_2: np.array mask for the source 2
    return the 2 sources separated (source1 source2)
    """
    if len(mix) != len(mask_1) or len(mix) != len(mask_2):
        raise Exception('The length of the mix and the masks should be the same')

    source1 = np.multiply(mix, mask_1)
    source2 = np.multiply(mix, mask_2)

    return source1, source2

