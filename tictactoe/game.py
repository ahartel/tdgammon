import unittest
from tictactoe.board import Board, Position, Move


class Game:
    PLAYER1 = Board.CROSS
    PLAYER2 = Board.CIRCL

    def __init__(self, board):
        self.board = board
        self.__num_moves = 0

    def is_finished(self):
        has_three_in_a_row = self.board.has_three_in_a_row() != 0
        no_fields_free = self.board.all_positions_full()
        return no_fields_free or has_three_in_a_row

    def get_winner(self):
        return self.board.has_three_in_a_row()

    def apply(self, move):
        if move.get_player() == self.board.CROSS:
            self.board.set_cross(move.get_pos())
        elif move.get_player() == self.board.CIRCL:
            self.board.set_circle(move.get_pos())
        else:
            raise NotImplementedError
        self.__num_moves += 1

    def roll_dice(self):
        return None

    def move(self, evaluation_function, player):
        possible_moves = []
        for row in range(Board.NUM_ROWS_AND_COLS):
            for col in range(Board.NUM_ROWS_AND_COLS):
                pos = Position(row, col)
                if self.board.is_empty(pos):
                    possible_moves.append(Move(pos, player))

        if len(possible_moves) == 0:
            print(self.board.positions)

        fun = lambda move : evaluation_function(self.board.get_network_inputs_with_move_applied(move))
        moves_values = list(map(fun, possible_moves))
        index_min = min(range(len(moves_values)), key=moves_values.__getitem__)

        return possible_moves[index_min]

    def get_num_moves(self):
        return self.__num_moves


class TTTGameTest(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board()
        self.game = Game(self.board)

    def test_NewGameIsOpen(self):
        self.assertEqual(self.game.is_finished(), False)

    def test_CanApplyMove(self):
        row = col = 1
        self.game.apply(self.board.CROSS, (row, col))
        self.assertEqual(self.game.is_finished(), False)

    def test_CanWinGame(self):
        row = col = 0
        self.game.apply(self.board.CROSS, (row, col))
        col += 1
        self.game.apply(self.board.CROSS, (row, col))
        self.assertEqual(self.game.is_finished(), False)
        col += 1
        self.game.apply(self.board.CROSS, (row, col))
        self.assertEqual(self.game.is_finished(), True)
        self.assertEqual(self.game.get_winner(), self.board.CROSS)

    def test_RollDiceReturnsNone(self):
        self.assertIsNone(self.game.roll_dice())

    def test_Argmin(self):
        values = [3,6,1,5]
        argmin = min(range(len(values)), key=values.__getitem__)
        self.assertEqual(argmin, 2)


if __name__ == '__main__':
    unittest.main()
