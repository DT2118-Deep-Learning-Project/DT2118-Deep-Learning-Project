#!/usr/bin/python
# source_separation.py
# 
import gen_input.filefeatures as sgf
import numpy as np
import neural_network.RNN
import neural_network.loss_function

setsize=10000
prefix="../../"

def train_srs():
    # Load train set    
    print 'Loading files'
    X = sgf.extract_fft(sgf.load_set_features(prefix + "features/tidigits_noisy/train", 0), 1)
    clean = sgf.extract_fft(sgf.load_set_features(prefix + "features/tidigits_clean/train", 0), 1)
    noise = sgf.extract_fft(sgf.load_set_features(prefix + "features/tidigits_noise/train", 0), 1)

    X = X[:setsize]
    Y = np.append(clean[:setsize], noise[:setsize], axis=1)

    scale = np.mean(Y)
    Y = Y / scale
    X = X / scale

    input_size = X.shape[1]

    # Create net
    print 'Building RNN'
    rnn = neural_network.RNN.RNN(input_size, 2, X, 2, loss=neural_network.loss_function.source_separation_loss_function)

    # Train net
    print 'Training'
    rnn.fit(X, Y, nb_epoch=2)

    # Save net
    print 'Saving'
    rnn.save()
    
    # Result
    print 'LOL'


def test_srs():
    # Load test set
    X = sgf.extract_fft(sgf.load_set_features(prefix + "features/tidigits_noisy/test", 0), 1)
    clean = sgf.extract_fft(sgf.load_set_features(prefix + "features/tidigits_clean/test", 0), 1)
    noise = sgf.extract_fft(sgf.load_set_features(prefix + "features/tidigits_noise/test", 0), 1)

    Y = np.append(clean, noise, axis=0)
    # Load net

    # Test net

    # Save output result

    # Result

print 'Train srs'
train_srs()
