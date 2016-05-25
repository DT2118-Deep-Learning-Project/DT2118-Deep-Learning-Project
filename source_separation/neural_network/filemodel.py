#!/usr/bin/python
"""
Function for loading/saving neural networks
"""

def save_model(model, name, overwrite = False):
    json_string = model.to_json()
    open(name + '.json', 'w').write(json_string)
    model.save_weights(name + 'weights.h5', overwrite)
    
def load_model(name):
    m = model_from_json(open(name + '.json').read())
    m.load_weights(name + '_weights.h5')
    return m
