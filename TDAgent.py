import numpy as np
from RandomAgent import RandomAgent
from board import Board


class TD0Agent (RandomAgent):
    def __init__(self, board, use_whites=False):
        super(TD0Agent, self).__init__(board, use_whites)

    def save_weights(self, filename):
        with open(filename, "wb") as file:
            np.save(file, self.mlp.hidden_weights)
            np.save(file, self.mlp.output_weights)
            np.save(file, self.mlp.hidden_biases)
            np.save(file, self.mlp.output_biases)

    def load_weights(self, filename):
        with open(filename, "rb") as file:
            hidden_weights = np.load(file)
            output_weights = np.load(file)
            self.mlp.set_weights(hidden_weights, output_weights)
            hidden_biases = np.load(file)
            output_biases = np.load(file)
            self.mlp.set_biases(hidden_biases, output_biases)

    def backprop(self, reward=None):
        if self.last_my_fields_before_move is None or self.last_other_fields_before_move is None:
            return

        if reward is None:
            inputs = Board.prepare_any_inputs(self.my_fields, self.other_fields)
            output_at_next_timestep = self.mlp.run_input(inputs, save_inputs_and_activations=False)
        else:
            output_at_next_timestep = reward

        inputs = Board.prepare_any_inputs(self.last_my_fields_before_move,
                                          self.last_other_fields_before_move)
        outputs = self.mlp.run_input(inputs)
        self.mlp.backprop(output_at_next_timestep)
