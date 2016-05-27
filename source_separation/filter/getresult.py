#!/usr/bin/python
"""
Set the environnement for playing with result
Parameter: 1: path to the features
"""
import numpy as np
import extractmask as em
import fileaudio as fa
import filter
import sys

reload(em)
reload(fa)
reload(filter)

train_noisy, train_noise, train_clean = em.getdata(sys.argv[1] , 'train', 0, 1)

clean_wav_data, clean_wav_name = fa.load_set_wav(sys.argv[1] + "tidigits_clean/train/man")
noise_wav_data, noise_wav_name = fa.load_set_wav(sys.argv[1] + "tidigits_noise/train/man")
noisy_wav_data, noisy_wav_name = fa.load_set_wav(sys.argv[1] + "tidigits_noisy/train/man")

