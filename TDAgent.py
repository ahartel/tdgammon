import numpy as np
from RandomAgent import RandomAgent
from tictactoe.board import Board


class TD0Agent (RandomAgent):
    gamma = 1.0
    _lambda = 0.5

    def __init__(self, board, num_hidden, player):
        super(TD0Agent, self).__init__(board, num_hidden, player)
        self.mlp.set_biases(np.zeros_like(self.mlp.hidden_biases),
                            np.zeros_like(self.mlp.output_biases))

        # eligibility trace
        self.et = None
        self.reset_trace()

        self.board_state_after_last_move = None

    def backprop(self, reward=None):
        if self.board_state_after_last_move is None:
            return

        if reward is None:
            inputs = Board.prepare_any_inputs(self.my_fields, self.other_fields)
            output_at_next_timestep = self.mlp.run_input(inputs, save_inputs_and_activations=False)
        else:
            output_at_next_timestep = reward

        inputs = Board.prepare_any_inputs(self.last_my_fields_before_move,
                                          self.last_other_fields_before_move)
        # outputs = self.mlp.run_input(inputs)
        self.mlp.backprop(output_at_next_timestep)

    def learn(self, reward=None):
        if self.board_state_after_last_move is None:
            return

        current_board_state = self.board.get_copy_of_state()

        inputs_old_state = Board.get_network_inputs_of_board_state(self.board_state_after_last_move)
        output_old_state = self.mlp.run_input(inputs_old_state, save_inputs_and_activations=True)

        inputs_new_state = Board.get_network_inputs_of_board_state(current_board_state)
        output_new_state = self.gamma * self.mlp.run_input(inputs_new_state, save_inputs_and_activations=False)
        if reward is not None:
            output_new_state += reward

        total_weight_update = [np.zeros_like(self.mlp.hidden_weights), np.zeros_like(self.mlp.output_weights)]
        # total_bias_update = [np.zeros_like(self.mlp.hidden_biases), np.zeros_like(self.mlp.output_biases)]

        assert(len(output_old_state) == 1)
        error_signal = output_new_state[0] - output_old_state[0]
        # print("Error signal for output_index {} is {}".format(output_index, error_signal))
        weight_updates, bias_updates = self.mlp.gradient()
        self.et[0] = weight_updates[0] + self.et[0] * self.gamma * self._lambda
        self.et[1] = weight_updates[1] + self.et[1] * self.gamma * self._lambda
        # self.e[output_index][2] = bias_updates[0] + self.e[output_index][2] * self.gamma * self._lambda
        # self.e[output_index][3] = bias_updates[1] + self.e[output_index][3] * self.gamma * self._lambda
        # print("Eligibility trace for output_index {} is {}".format(output_index, self.e[output_index]))
        total_weight_update[0] += self.et[0] * error_signal
        total_weight_update[1] += self.et[1] * error_signal
        # total_bias_update[0] += self.e[output_index][2] * error_signal
        # total_bias_update[1] += self.e[output_index][3] * error_signal

        self.mlp.add_to_weights(total_weight_update)
        # self.mlp.add_to_biases(total_bias_update)

    def reset_trace(self):
        self.et = [np.zeros_like(self.mlp.hidden_weights),
                   np.zeros_like(self.mlp.output_weights),
                   np.zeros_like(self.mlp.hidden_biases),
                   np.zeros_like(self.mlp.output_biases)]

    def move(self, dice):
        return RandomAgent.move(self, dice)

    def remember_board_state(self):
        self.board_state_after_last_move = self.board.get_copy_of_state()

