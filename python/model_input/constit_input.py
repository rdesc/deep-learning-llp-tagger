from model_input import ModelInput


class ConstitInput(ModelInput):

    def __init__(self, rows_max, num_features, layers_cnn, nodes_lstm):
        ModelInput.__init__(self, rows_max, num_features, layers_cnn, nodes_lstm)

