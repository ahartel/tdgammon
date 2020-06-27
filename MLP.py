import numpy as np
import math


class MLP:

    def __init__(self, num_inputs, num_hiddens, num_outputs):
        self.input_weights = np.random.standard_normal((num_hiddens, num_inputs))
        self.output_weights = np.random.standard_normal((num_outputs, num_hiddens))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def run_input(self, inputs):
        hidden = self.sigmoid(self.input_weights.dot(inputs))
        outputs = self.sigmoid(self.output_weights.dot(hidden))
        return outputs
