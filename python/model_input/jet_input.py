from model_input.model_input import ModelInput


class JetInput(ModelInput):

    def __init__(self, rows_max=0, num_features=0, layers_cnn=0, nodes_lstm=0):
        ModelInput.__init__(self, rows_max, num_features, layers_cnn, nodes_lstm)

    def extract_and_split_data(self, X_train, X_test, X_val, start, end):
        train = X_train.loc[:, start:end]
        test = X_test.loc[:, start:end]
        val = X_val.loc[:, start:end]

        # print some details
        print("\nShape: %.0f" % (train.shape[1]))
        print("Number of training examples %.0f" % (train.shape[0]))
        print("Number of testing examples %.0f" % (test.shape[0]))
        print("Number of validating examples %.0f\n" % (val.shape[0]))

        return train, test, val
