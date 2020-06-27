import MLP


class RandomAgent:
    def __init__(self, board):
        self.board = board
        self.mlp = MLP.MLP((self.board.NUM_FIELDS * 2 * 4) + 2 + 2, 40, 4)
        inputs_to_mlp = self.board.prepare_inputs()
        print(self.mlp.run_input(inputs_to_mlp))

    def move(self, dice):
        return [[24, 23], [24, 23], [23, 22], [23, 22]]
