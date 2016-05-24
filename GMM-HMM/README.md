Contain all the tools for using a GMM-HMM model for doing ASR on the tidigits.a
The recognition is done with MFCC_0_D_A (MFCC with energy, derivative and acceleration)
The GMM-HMM model is based on HTK.

# Utilisation
1. Create workdir with informations on the utterances
$> tools/prepare_data.sh $tidigits_type

With $tidigits_type = {tidigits_clean, tidigits_noise, tidigits_noisy}

2. Train the GMM-HMM model
$> tools/train_gmm-hmm.sh MFCC_0_D_A

3. Test the model
$> tools/test_gmm-hmm.sh MFCC_0_D_A
