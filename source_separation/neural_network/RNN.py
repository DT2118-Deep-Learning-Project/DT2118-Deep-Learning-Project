import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import SimpleRNN, Dense
from output_layer import Output_Layer
from mask_data_callback import Mask_Data_Callback
from loss_function import source_separation_loss_function

class RNN:
    def __init__(self, input_size, hidden_layer, nodes_hl, stfs, timesteps,
                 optimizer='sgd', loss='mse', activation='relu'):
        self.stfs = stfs
        self.model = self.build(input_size, hidden_layer, nodes_hl, timesteps, activation)
        self.model.compile(optimizer=optimizer, loss=loss)
        self.name = "model_rnn_" + activation + "_" + str(hidden_layer) + "_" + str(nodes_hl)

    def build(self, input_size, hidden_layer, nodes_hl, timesteps, activation):
        '''
            The Recurrent Neural Network from the Monaural Speech
            Recognition paper

            :param input_size: Size of the input layer
            :param hidden_layer: Number of hidden layers
            :param stfs: The STFs as a numpy array with shape: (SET_SIZE, OUTPUT_SIZE)
            :param timesteps: Recurrence depth
        '''
        output_size = self.stfs.shape[1]

        model = Sequential()
        model.add(SimpleRNN(nodes_hl, input_shape=(timesteps, input_size),
                return_sequences=True, activation=activation))

        for i in range(hidden_layer):
            model.add(SimpleRNN(nodes_hl, return_sequences=True, activation=activation))

        model.add(SimpleRNN(2 * output_size, return_sequences=False, activation=activation))
        model.add(Output_Layer(output_size, self.stfs))
        return model

    def prepare_data(self, X_raw, y_raw=None, n_prev=2):  
        X = [] 
        for i in range(X_raw.shape[0] - n_prev + 1):
            X.append(X_raw[i:i+n_prev,:])
        X = np.array(X)
        if y_raw is None:
            return X
        else: 
            y = np.array(y_raw[n_prev - 1:])
            return X,y
        
    def fit(self, noisy, targets, nb_epoch=10, batch_size=1):
        X, y = self.prepare_data(noisy, targets)
        mask_data = Mask_Data_Callback(self.stfs.shape[0])
        self.model.fit(X, y, nb_epoch=nb_epoch, batch_size=1, callbacks=[mask_data])

    def save(self, name='model', overwrite=False):
        json_string = self.model.to_json()
        open(self.name + '.json', 'w').write(json_string)
        self.model.save_weights(self.name + '_weights.h5', overwrite)

    def load_weight(self, path):
      #m = model_from_json(open(name + '.json').read())
      self.model.load_weights(path)

    def evaluate(self, x, y, batch_size=1):
        return self.model.evaluate(x, y, batch_size=batch_size)

    def predict(self, x, batch_size=1):
        return self.model.predict(x,batch_size=batch_size)

if __name__ == '__main__':
    # Dummy data
    INPUT_SIZE  = 1024
    OUTPUT_SIZE = 2 * INPUT_SIZE
    SET_SIZE    = 1000

    clean = np.random.random((SET_SIZE, INPUT_SIZE))
    noise = 0.10 * np.random.random((SET_SIZE, INPUT_SIZE))
    noisy  = clean + noise
    target = np.append(clean, noise, axis=1)

    print target.shape
    print noisy.shape

    rnn = RNN(INPUT_SIZE, 2, 150, noisy, 2, loss=source_separation_loss_function)
    rnn.fit(noisy, target)
