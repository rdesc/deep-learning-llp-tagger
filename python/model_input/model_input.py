from keras.layers import Input, Conv1D, CuDNNLSTM, Dense
from keras.regularizers import L1L2


class ModelInput:

    def __init__(self, name, rows_max, num_features, layers_cnn, nodes_lstm):
        self.name = name
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

    def init_keras_cnn_input_output(self, shape, activation='relu'):
        # input into first cnn layer
        input_tensor = Input(shape=shape, dtype='float32', name=self.name+'_input')

        # init output
        output_tensor = Conv1D(filters=self.layers_cnn.pop(0), kernel_size=1, activation=activation, input_shape=shape)(
            input_tensor)

        for i in range(len(self.layers_cnn)):
            if len(self.layers_cnn) == 1:
                output_tensor = Conv1D(filters=self.layers_cnn.pop(0), kernel_size=1, activation=activation,
                                       name=self.name+'_final_conv1d')(output_tensor)
            else:
                output_tensor = Conv1D(filters=self.layers_cnn.pop(0), kernel_size=1, activation=activation)(output_tensor)

        return input_tensor, output_tensor

    def init_keras_lstm(self, reg_value, input_tensor, activation='softmax'):
        output_tensor = CuDNNLSTM(self.nodes_lstm, kernel_regularizer=L1L2(l1=reg_value, l2=reg_value))(input_tensor)

        # Dense layer tracks performances of LSTM
        return Dense(3, activation=activation, name=self.name+'_output')(output_tensor)
