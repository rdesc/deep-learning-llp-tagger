from model_input import ModelInput


class JetInput(ModelInput):

    def __init__(self, rows_max=0, num_features=0, layers_cnn=0, nodes_lstm=0):
        ModelInput.__init__(self, rows_max, num_features, layers_cnn, nodes_lstm)

    def extract_and_split_data(self, X_train, X_test, X_val, start, end):
        train = X_train.loc[:, start:end]
        train = train.values.reshape(train.shape[0], self.rows_max, self.num_features)
        test = X_test.loc[:, start:end]
        test = test.values.reshape(test.shape[0], self.rows_max, self.num_features)
        val = X_val.loc[:, start:end]
        val = val.values.reshape(val.shape[0], self.rows_max, self.num_features)

        # print some details
        print("\nShape: %.0f x %.0f" % (train.shape[1], train.shape[2]))
        # TODO: remove below 2 lines
        print("Shape: %.0f x %.0f" % (test.shape[1], test.shape[2]))
        print("Shape: %.0f x %.0f" % (val.shape[1], val.shape[2]))
        print("Number of training examples %.0f" % (train.shape[0]))
        print("Number of testing examples %.0f" % (test.shape[0]))
        print("Number of validating examples %.0f\n" % (val.shape[0]))

        return train, test, val
