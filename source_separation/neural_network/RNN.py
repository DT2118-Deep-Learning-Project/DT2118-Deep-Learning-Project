import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import SimpleRNN, Dense
from output_layer import Output_Layer
from mask_data_callback import Mask_Data_Callback
from loss_function import source_separation_loss_function

class RNN:
    def __init__(self, input_size, hidden_layer, stfs, timesteps,
                 optimizer='sgd', loss='mse'):
        self.stfs = stfs
        self.model = self.build(input_size, hidden_layer, timesteps)
        self.model.compile(optimizer=optimizer, loss=loss)

    def build(self, input_size, hidden_layer, timesteps):
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
        model.add(SimpleRNN(output_size, input_shape=(timesteps, input_size),
                return_sequences=True, activation='relu'))

        for i in range(hidden_layer):
            model.add(SimpleRNN(150, return_sequences=True, activation = 'relu'))

        model.add(SimpleRNN(2 * output_size, return_sequences=False, activation = 'relu'))
        model.add(Output_Layer(output_size, self.stfs))
        return model

    def prepare_data(self, X_raw, y_raw, n_prev=2):  
        X = [] 
        for i in range(X_raw.shape[0] - n_prev + 1):
            X.append(X_raw[i:i+n_prev,:])
        X = np.array(X)
        y = np.array(y_raw[n_prev - 1:])
        return X,y
        
    def fit(self, noisy, targets, nb_epoch=10, batch_size=1):
        X, y = self.prepare_data(noisy, targets)
        mask_data = Mask_Data_Callback(self.stfs.shape[0])
        self.model.fit(X, y, nb_epoch=10, batch_size=1, callbacks=[mask_data])

    def save(self, name, overwrite=False):
        json_string = self.model.to_json()
        open(name + '.json', 'w').write(json_string)
        self.model.save_weights(name + 'weights.h5', overwrite)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def predict(self, x, y):
        return self.model.predict(x, y)

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

    rnn = RNN(INPUT_SIZE, 2, noisy, 2, loss=source_separation_loss_function)
    rnn.fit(noisy, target)
