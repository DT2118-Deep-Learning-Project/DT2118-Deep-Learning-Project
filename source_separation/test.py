#!/usr/bin/python
# source_separation.py
# 
import source_separation.gen_input.filefeatures as sgf
import numpy as np
from source_separation.neural_network import *

setsize=10000
INPUTSIZE=1024

def train_srs():
    # Load train set    
    X = sgf.extract_fft(sgf.load_set_features("../data/features/tidigits_noisy/train", 0), 1)
    clean = sgf.extract_fft(sgf.load_set_features("../data/features/tidigits_clean/train", 0), 1)
    noise = sgf.extract_fft(sgf.load_set_features("../data/features/tidigits_noise/train", 0), 1)
    Y = np.append(clean[:setsize], noise[:setsize], axis=0)
    # Create net
    rnn = neural_network.RNN(INPUT_SIZE, 2, noisy, 2, loss=source_separation_loss_function)

    # Train net
    rnn.fit(X[:setsize], Y)

    # Save net
    rnn.save()
    
    # Result
    print 'LOL'


def test_srs():
    # Load test set
    X = sgf.extract_fft(sgf.load_set_features("../data/features/tidigits_noisy/test", 0), 1)
    clean = sgf.extract_fft(sgf.load_set_features("../data/features/tidigits_clean/test", 0), 1)
    noise = sgf.extract_fft(sgf.load_set_features("../data/features/tidigits_noise/test", 0), 1)
    Y = np.append(clean, noise, axis=0)
    # Load net

    # Test net

    # Save output result

    # Result
