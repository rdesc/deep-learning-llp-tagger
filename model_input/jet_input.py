from keras.layers import Input, Dense
from model_input.model_input import ModelInput


class JetInput(ModelInput):
    """
        This is a subclass for jet input to the model
    """

    def __init__(self, name, input_count=0, vars_count=0, filters_cnn=0, nodes_lstm=0):
        ModelInput.__init__(self, name, input_count, vars_count, filters_cnn, nodes_lstm)

    def extract_and_split_data(self, X_train, X_val, X_test, start, end):
        train = X_train.loc[:, start:end]
        val = X_val.loc[:, start:end]
        test = X_test.loc[:, start:end]

        # print some details
        print("Shape: %.0f" % (train.shape[1]))
        print("Number of training examples %.0f" % (train.shape[0]))
        print("Number of validating examples %.0f" % (val.shape[0]))
        print("Number of testing examples %.0f" % (test.shape[0]))

        return train, val, test

    def init_keras_dense_input_output(self, shape, activation='softmax'):
        # input layer into keras model
        input_tensor = Input(shape=shape, dtype='float32', name=self.name+'_input')

        # setup up Dense layer
        output_tensor = Dense(3)(input_tensor)
        output_tensor = Dense(3, activation=activation, name=self.name+'_output')(output_tensor)

        return input_tensor, output_tensor
