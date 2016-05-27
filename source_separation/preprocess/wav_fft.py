# -*- coding: utf-8 -*-
"""
    This file can load WAV files, convert samples to STFT / FFT,
    and writes features data to be later used again
"""
import scipy.io.wavfile
import numpy as np

FRAMESZ = 0.0256 # window length in ms
HOP     = 0.0128 # window interval in ms
FS      = 20000 # sample rate

def stft(x, fs, framesz=FRAMESZ, hop=HOP):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hanning(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp])
        for i in range(0, len(x)-framesamp, hopsamp)]) # We lose framesamp samples... 
    return np.complex64(X) # reduce precision from complex 128
 
def istft(X, fs, hop=HOP):
    nbrsamp, framesamp = X.shape
    x = scipy.zeros(nbrsamp*framesamp*hop/FRAMESZ)
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    return np.int32(x)

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
    return ft

def writeWAV(path, stft_data):
    """
        Convert the stft into wav samples and write it to the specified path
    """
    T = stft_data.shape[0] * HOP
    print T
    samples = istft(stft_data, FS, HOP)
    print samples.shape
    scipy.io.wavfile.write(path, FS, samples)

def writeFFT(path, filename, stft_data):
    """
        Save STFTs into a .npz file
    """
    np.savez(path + '/' + filename + '_fft.npz', filename=filename, fft=stft_data)

