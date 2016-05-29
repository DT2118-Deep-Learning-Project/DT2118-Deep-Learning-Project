#!/usr/bin/python
"""
 Run this script for generating all the fft of the sources
 Note:
    You need to source the file config_env at the root of the repository
    The original wav files have to be in (PROJ_DIR)/data with
        tidigits_clean
        tidigits_noise
        tidigits_noisy
    The extracted fft will be in the directory (PROJ_DIR)/data/features
"""

import source_separation.preprocess.dataIO as dataIO
import os

# Raise an error if don't exist --> source config_env
target_dir=os.environ['SPEECH_PROJ_DIR']

dataIO.extractAllFFTfromWAV(target_dir+"/data", target_dir+"/data/features", percent=0.2)

