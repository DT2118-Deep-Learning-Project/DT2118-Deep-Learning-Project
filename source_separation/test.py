#!/usr/bin/python
# source_separation.py
# 
import gen_input.filefeatures as sgf
import gen_input.invfeatures as invf
import numpy as np
import neural_network.RNN
import neural_network.loss_function

setsize=150
prefix="../data/"
hidden_layer = 2
nodes = 150
activation='relu'

def load_dataset(datatype):
    if datatype != 'train' or datatype != 'test':
        raise Exception('Please pass parameter "train" or "test" to the load data function')

    X     = sgf.extract_fft(sgf.load_set_features(prefix + "features/tidigits_noisy/" + datatype, 0), 1)
    clean = sgf.extract_fft(sgf.load_set_features(prefix + "features/tidigits_clean/" + datatype, 0), 1)
    noise = sgf.extract_fft(sgf.load_set_features(prefix + "features/tidigits_noise/" + datatype, 0), 1)

    X = X[:setsize]
    Y = np.append(clean[:setsize], noise[:setsize], axis=1)

    # Scaling to get values easier to handle
    scale = np.mean(Y)
    Y     = Y / scale
    X     = X / scale

    input_size = X.shape[1]
    return X, Y

def save_to_wav(name, stft):
    '''
        :param stft: must be an 2D-array of size (N_windows, STFT_length)
    '''
    samples = invf.fromffttowav(data)
    fs      = 20000
    path    = '../data/output' + name
    print "Saving wave file to: ", path
    scipy.io.wavfile.write(path, fs, samples)

def train_srs():
    # Load train set    
    print 'Loading files'
    X, Y = load_dataset('train')

    # Create net
    print 'Building RNN'
    rnn = neural_network.RNN.RNN(input_size, hidden_layer, nodes, X, 2,
            loss=neural_network.loss_function.source_separation_loss_function, activation=activation)

    # Train net
    print 'Training'
    rnn.fit(X, Y, nb_epoch=2)

    # Save net
    print 'Saving'
    rnn.save()
    
    # Result
    print 'LOL'
    return rnn


def test_srs():
    # Load test set
    print 'Loading files'
    X, Y = load_dataset('test')

    # Load net

    # Test net

    # Save output result

    # Result
print 'Train srs'
rnn = train_srs()

X = sgf.extract_fft(sgf.load_set_features(prefix + "features/tidigits_noisy/test", 0), 1)
pred = rnn.predict(rnn.prepare_data(X[:150]))


