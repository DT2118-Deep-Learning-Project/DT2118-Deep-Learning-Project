#!/usr/bin/python
"""
    Display data
"""
import scipy
import numpy as np
import matplotlib.pyplot as plt

def complexshow(input):
    input = scipy.absolute(input)
    logshow(input)

def logshow(input):
    input = np.log(input)
    show(input)

def show(input):
    plt.clf()
    plt.imshow(input.transpose(), origin='lower', interpolation='nearest', aspect='auto')
    plt.colorbar()    
    plt.show()
    
def showweights(nn):
    i=0
    for l in nn.model.layers:
        i=i+1
        w = l.get_weights()
        if(len(w) == 0):
            continue
        # Print W
        plt.figure(1)
        show(w[0])
        #Print Bias
        plt.figure(2)
        plt.plot(w[1])
        print("End " + str(i))
