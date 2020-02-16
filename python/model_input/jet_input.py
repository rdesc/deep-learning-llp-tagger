from keras.layers import Input, Dense
from model_input.model_input import ModelInput


class JetInput(ModelInput):

    def __init__(self, name, rows_max=0, num_features=0, filters_cnn=0, nodes_lstm=0):
        ModelInput.__init__(self, name, rows_max, num_features, filters_cnn, nodes_lstm)

    def extract_and_split_data(self, X_train, X_test, X_val, start, end):
        train = X_train.loc[:, start:end]
        test = X_test.loc[:, start:end]
        val = X_val.loc[:, start:end]

        # print some details
        print("Shape: %.0f" % (train.shape[1]))
        print("Number of training examples %.0f" % (train.shape[0]))
        print("Number of testing examples %.0f" % (test.shape[0]))
        print("Number of validating examples %.0f" % (val.shape[0]))

        return train, test, val

    def init_keras_dense_input_output(self, shape, activation='softmax'):
        # input layer into keras model
        input_tensor = Input(shape=shape, dtype='float32', name=self.name+'_input')

        # setup up Dense layer
        output_tensor = Dense(3)(input_tensor)
        output_tensor = Dense(3, activation=activation, name=self.name+'_output')(output_tensor)

        return input_tensor, output_tensor
