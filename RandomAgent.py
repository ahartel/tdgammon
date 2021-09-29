import random

import MLP
import copy
import numpy as np


class RandomAgent:
    def __init__(self, board, num_hidden, player):
        # if use_whites:
        #     self.my_fields = board.whites
        #     self.other_fields = board.blacks
        # else:
        #     self.my_fields = board.blacks
        #     self.other_fields = board.whites
        self.__player = player
        self.mlp = MLP.RandomThreeLayerMLP(board.get_network_input_size(), num_hidden, num_outputs=1, learning_rate=0.5)
        self.last_my_fields_before_move = None
        self.last_other_fields_before_move = None
        self.board = board

    def evaluate_moves_by_mlp(self, inputs):
        # old code for TD Gammon
        # max_move_index = 0
        # max_move_value = 0
        # for idx, move_die in enumerate(moves):
        #     die, move = move_die
        #     my_fields_after = copy.copy(my_fields)
        #     other_fields_after = copy.copy(other_fields)
        #     try:
        #         Game.apply_move(my_fields_after, other_fields_after, move)
        #         inputs = Board.prepare_any_inputs(my_fields_after, other_fields_after)
        #         outputs = self.mlp.run_input(inputs)
        #         if outputs[0] > max_move_value:
        #             max_move_value = max_move_value
        #             max_move_index = idx
        #     except Exception as e:
        #         continue

        outputs = self.mlp.run_input(inputs, save_inputs_and_activations=False)
        return outputs[0]

    def print_intermediate_board(self, my_fields, other_fields):
        print("Intermediate board:")
        if self.__use_whites:
            Board(my_fields, other_fields).print()
        else:
            Board(other_fields, my_fields).print()

    def move(self, game):
        return game.move(self.evaluate_moves_by_mlp, self.__player)

    @staticmethod
    def reset_trace():
        pass

    def learn(self, reward):
        pass

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

    def remember_board_state(self):
        pass


class RandomTTTAgent(RandomAgent):
    def __init__(self, board, num_hidden, player):
        super(RandomTTTAgent, self).__init__(board, num_hidden, player)

    def evaluate_moves_by_mlp(self, inputs):
        return random.uniform(0, 1)

