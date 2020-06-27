import sys
import MLP
from board import Board


if __name__ == '__main__':
    board = Board()
    board.print()
    mlp = MLP.MLP((board.NUM_FIELDS * 2 * 4) + 2 + 2, 40, 4)
    inputs_to_mlp = board.prepare_inputs()
    print(mlp.run_input(inputs_to_mlp))
