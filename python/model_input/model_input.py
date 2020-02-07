from keras.layers import Input, Conv1D


class ModelInput:

    def __init__(self, rows_max, num_features, layers_cnn, nodes_lstm):
        self.rows_max = rows_max
        self.num_features = num_features
        self.layers_cnn = layers_cnn
        self.nodes_lstm = nodes_lstm

    def extract_and_split_data(self, X_train, X_test, X_val, start, end):
        train = X_train.loc[:, start:end + str(self.rows_max - 1)]
        train = train.values.reshape(train.shape[0], self.rows_max, self.num_features)
        test = X_test.loc[:, start:end + str(self.rows_max - 1)]
        test = test.values.reshape(test.shape[0], self.rows_max, self.num_features)
        val = X_val.loc[:, start:end + str(self.rows_max - 1)]
        val = val.values.reshape(val.shape[0], self.rows_max, self.num_features)

        # print some details
        print("Shape: %.0f x %.0f" % (train.shape[1], train.shape[2]))
        print("Number of training examples %.0f" % (train.shape[0]))
        print("Number of testing examples %.0f" % (test.shape[0]))
        print("Number of validating examples %.0f" % (val.shape[0]))

        return train, test, val

    def init_keras_cnn_input_output(self, shape, name, activation='relu'):
        # input into first cnn layer
        input_tensor = Input(shape=shape, dtype='float32', name=name)

        # init output
        output_tensor = Conv1D(filters=self.layers_cnn.pop(0), kernel_size=1, activation=activation, input_shape=shape)(
            input_tensor)

        for i in range(len(self.layers_cnn)):
            output_tensor = Conv1D(filters=self.layers_cnn.pop(0), kernel_size=1, activation=activation)(output_tensor)

        return input_tensor, output_tensor
