from keras.layers import Input, Conv1D, CuDNNLSTM, Dense
from keras.regularizers import L1L2


class ModelInput:
    """
        This is a class for inputs to the model i.e. constit, track, MSeg, jet (subclass)

        Attributes:
            name (str): name of input
            rows_max (int): number of inputs (rows in input data variable X)
            num_features (int): number of features or variables (columns in input data variable X)
            filters_cnn (list): list of number of filters for each Conv1D layer
            nodes_lstm (int): number of nodes for lstm layer
    """

    def __init__(self, name, rows_max, num_features, filters_cnn=0, nodes_lstm=0):
        self.name = name
        self.rows_max = rows_max
        self.num_features = num_features
        self.filters_cnn = filters_cnn
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

    def init_keras_layers(self, shape, reg_value, activation_cnn='relu', activation_lstm='softmax'):
        # input to first model layer
        input_tensor = Input(shape=shape, dtype='float32', name=self.name + '_input')
        output_tensor = None
        dense_tensor = None

        # check if model input has conv1d layers
        if self.filters_cnn:
            # init output
            output_tensor = Conv1D(filters=self.filters_cnn[0], kernel_size=1, activation=activation_cnn,
                                   input_shape=shape)(input_tensor)

            # iterate over conv1d layers
            for filters in self.filters_cnn[1:]:
                # add name to final layer
                if filters == self.filters_cnn[-1]:
                    output_tensor = Conv1D(filters=filters, kernel_size=1, activation=activation_cnn,
                                           name=self.name + '_final_conv1d')(output_tensor)
                else:
                    output_tensor = Conv1D(filters=filters, kernel_size=1, activation=activation_cnn)(output_tensor)

        # check if model input has an lstm layer
        if self.nodes_lstm:
            output_tensor = CuDNNLSTM(self.nodes_lstm, kernel_regularizer=L1L2(l1=reg_value, l2=reg_value)) \
                (output_tensor if output_tensor else input_tensor)
            # Dense layer tracks performances of LSTM
            dense_tensor = Dense(3, activation=activation_lstm, name=self.name + '_output')(output_tensor)

        if not (self.nodes_lstm and self.filters_cnn):
            print("\nNo Conv1D or LSTM layers in model architecture!\n")
            # set output tensor equal to input tensor
            output_tensor = input_tensor

        return input_tensor, output_tensor, dense_tensor
