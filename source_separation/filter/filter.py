#!/usr/bin/python
# filter.py
# Function for applying a mask on a stft for splitting it in two parts
import numpy as np

def apply_filter( mix, mask_1, mask_2 ):
    """
    mix: np.array sound with source 1 and 2 mixed
    mask_1: np.array mask for the source 1
    mask_2: np.array mask for the source 2
    return the 2 sources separated (source1 source2)
    """
    if len(mix) != len(mask_1) or len(mix) != len(mask_2):
        raise Exception('The length of the mix and the masks should be the same')

    source1 = np.multiply(mix, mask_1)
    source2 = np.multiply(mix, mask_2)

    return source1, source2

