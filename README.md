# DT2118-Deep-Learning-Project


## Generate the WAV

To generate the noisy sound files, put the tidigits directory into the `generate_data/` folder, and from `generate_data/` run `make`.
You will need `sox` to do that ! (`apt-get install sox`, `yum install sox`, and so on...)

## Generate the datasets

The dataset is composed of the STFT of the clean / noise / noisy signal. 
Sound processing is done in the `source_separation/wav_fft.py` file, and the dataset is built in `dataIO.py`.

To det the train and test data, from the `source_separation` directory just do:
```
from preprocessing import dataIO

Xtrain, Ytrain = dataIO.train_set()
Xtest, Ytest   = dataIO.test_set()
```

Note that the `train_set()` function returns one big 2D array of all the STFT from all the files concatenated. 

But the `test_set()` returns a list, for which each element contains the STFT for one WAV file. 
This enables to test the network only on a few files.

