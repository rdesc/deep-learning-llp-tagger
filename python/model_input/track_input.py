from model_input.model_input import ModelInput


class TrackInput(ModelInput):

    def __init__(self, rows_max, num_features, layers_cnn, nodes_lstm):
        ModelInput.__init__(self, rows_max, num_features, layers_cnn, nodes_lstm)

