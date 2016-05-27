#!/usr/bin/python
# filter.py
# Function for applying a mask on a stft for splitting it in two parts
import numpy as np
from source_separation.bss_eval import separation as sp

def get_bss_result(s1, s2, es1, es2):
    """
    s1, s2: the targeted sources
    es1, es2: the esperated sources
    Return the (SDR, SIR, SAR)
    """
    size = min( len(s1), len(s2), len(es1), len(es2) )

    sdr, sir, sar, perm = sp.bss_eval_sources(np.array( [s1[1:size],  s2[1:size]] ), 
                        np.array( [es1[1:size], es2[1:size]] ))

    return sdr, sir, sar

def batch_bss_result(s1, s2, es1, es2):
    """
    Compute BSS stat on a set of samples
    input
    matrix nxm with n number of samples and m length
    return
    vector of len n for 
    sdr, sir, sar
    """
    n = s1.size
    sdr = []
    sir = []
    sar = []

    for i in range(n):
        t1, t2, t3 = get_bss_result( s1[i], s2[i], es1[i], es2[i] )
        sdr.append(t1)
        sir.append(t2)
        sar.append(t3)

    return np.array(sdr), np.array(sir), np.array(sar)



