from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

# SEE: https://github.com/fchollet/keras/issues/622 

class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        if input_dim != self.output_dim:
            raise Exception('GROS FAIL... Input and output size are different!') 
        initial_weight_value = np.diag((input_dim, input_dim))
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

