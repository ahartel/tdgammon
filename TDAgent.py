import numpy as np
from RandomAgent import RandomAgent
from backgammon.board import Board


class TD0Agent (RandomAgent):
    gamma = 0.1

    def __init__(self, board, use_whites=False):
        super(TD0Agent, self).__init__(board, use_whites)
        self.mlp.set_biases(np.zeros_like(self.mlp.hidden_biases),
                            np.zeros_like(self.mlp.output_biases))

        self.e = [[np.zeros_like(self.mlp.hidden_weights),
                   np.zeros_like(self.mlp.output_weights),
                   np.zeros_like(self.mlp.hidden_biases),
                   np.zeros_like(self.mlp.output_biases)] for _ in range(self.mlp.num_outputs)]

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

    def learn(self, reward=None):
        if self.last_my_fields_before_move is None or self.last_other_fields_before_move is None:
            return

        if reward is None:
            inputs_new_state = Board.prepare_any_inputs(self.my_fields, self.other_fields)
            output_new_state = self.mlp.run_input(inputs_new_state, save_inputs_and_activations=False)
        else:
            output_new_state = reward

        inputs = Board.prepare_any_inputs(self.last_my_fields_before_move,
                                          self.last_other_fields_before_move)
        outputs = self.mlp.run_input(inputs)

        total_weight_update = [np.zeros_like(self.mlp.hidden_weights), np.zeros_like(self.mlp.output_weights)]
        #total_bias_update = [np.zeros_like(self.mlp.hidden_biases), np.zeros_like(self.mlp.output_biases)]
        for output_index in range(self.mlp.num_outputs):
            error_signal = output_new_state[output_index] - outputs[output_index]
            #print("Error signal for output_index {} is {}".format(output_index, error_signal))
            weight_updates, bias_updates = self.mlp.gradient(output_index)
            self.e[output_index][0] = weight_updates[0] + self.e[output_index][0] * self.gamma
            self.e[output_index][1] = weight_updates[1] + self.e[output_index][1] * self.gamma
            #self.e[output_index][2] = bias_updates[0] + self.e[output_index][2] * self.gamma
            #self.e[output_index][3] = bias_updates[1] + self.e[output_index][3] * self.gamma
            #print("Eligibility trace for output_index {} is {}".format(output_index, self.e[output_index]))
            total_weight_update[0] += self.e[output_index][0] * error_signal
            total_weight_update[1] += self.e[output_index][1] * error_signal
            #total_bias_update[0] += self.e[output_index][2] * error_signal
            #total_bias_update[1] += self.e[output_index][3] * error_signal

        self.mlp.add_to_weights(total_weight_update)
        #self.mlp.add_to_biases(total_bias_update)

    def reset_trace(self):
        self.e = self.e = [[np.zeros_like(self.mlp.hidden_weights),
                   np.zeros_like(self.mlp.output_weights),
                   np.zeros_like(self.mlp.hidden_biases),
                   np.zeros_like(self.mlp.output_biases)] for _ in range(self.mlp.num_outputs)]