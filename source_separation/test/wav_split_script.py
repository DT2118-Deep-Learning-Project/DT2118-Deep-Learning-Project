#!/usr/bin/python
"""
 Test the RNN fos splitting a wav
"""

import source_separation.preprocess.dataIO as dataIO
import source_separation.preprocess.wav_fft as wav_fft
import source_separation.test.testnet as testnet
import scipy
import numpy as np

X_test, Y_test = dataIO.test_set()
X_train, Y_train = dataIO.train_set()

mix_stft = X_train[:180, :]
mix_stft_en = scipy.absolute(mix_stft)
tar_stft = Y_train[:180, :]
tar_stft_en = scipy.absolute(tar_stft)
#scale = 2.31817e+09
scale = np.mean(mix_stft_en)
var = np.std(mix_stft_en)
print("Scale: " + str(scale))
mix_stft_en = (mix_stft_en-scale)/var
tar_stft_en = (tar_stft_en-scale)/var

# Retrieve rnn
#rnn= testnet.retrievernn("results/model_rnn_relu_2_150_weights.h5", 2, 150, 'relu', mix_stft_en)
dnn= testnet.retrievednn("results/model_dnn_relu_2_450_weights.h5", 2, 450, 'relu', mix_stft_en)

# Only for RNN
#input = rnn. prepare_data(mix_stft_en)
#pred_stft_en = rnn.predict(input)

pred_stft_en = dnn.predict(mix_stft_en)

# Retrieve mask and apply it 
# Maybe wrong because there are some weights in the last layer, need to verify...
# For RNN...
#mask_1 = pred_stft_en[:, :512] / mix_stft_en[:186]
#pred_stft = mix_stft[:186]* mask_1

mask_1 = pred_stft_en[:, :512] / mix_stft_en
pred_stft = mix_stft * mask_1

wav_fft.writeWAV("essai", pred_stft)


