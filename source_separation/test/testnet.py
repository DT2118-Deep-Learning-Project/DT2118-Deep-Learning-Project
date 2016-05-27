#!/usr/bin/python

import numpy as np
import source_separation.neural_network.RNN
import source_separation.neural_network.loss_function
import filter as fi
import fileaudio as fa

def retrievenn(path, hidden_layer, nodes, activation, input_size=512):
    """
    Initiate a NN and load the weights at the end of the training
    path: point to the file where the weights are saved
    """
    rnn = source_separation.neural_network.RNN.RNN(input_size, hidden_layer, nodes, np.zeros((2, 512)), 2,
       loss=source_separation.neural_network.loss_function.source_separation_loss_function, activation=activation)

    rnn.load_weight(path)
    return rnn

def getresult(rnn, noisy_fft, clean_wav, noise_wav, name):
    # Coupute source separation
    noisy_fft, y = rnn. prepare_data(noisy_fft, np.zeros(10))
    out = rnn.predict(noisy_fft)
    n, m = out.shape
    pred_clean_fft, pred_noise_fft = out[:, :m/2], out[:, m/2:]

    # Inverse fft
    #pred_clean_wav = pred_clean_fft
    #pred_noise_wav = pred_noise_fft

    # Compute bss
    sdr, sir, sar = fi.filter.get_bss_result(clean_wav, noise_wav, pred_clean_wav, pred_noise_wav)

    # Print bss
    print("SDR: " + str(sdr) + "\n"
            "SIR: " + str(sir) + "\n"
            "SAR: " + str(sar))

    # Write wav
    fa.save_set_wav(".", [name+"_clean", name+"_noise"], [pred_clean_wav, pred_noise_wav])

    return pred_clean_wav, pred_noise_wav, sdr, sir, sar
    


