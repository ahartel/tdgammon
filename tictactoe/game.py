import unittest
from tictactoe.board import Board

class Game:
    def __init__(self, board):
        self.board = board

    def is_finished(self):
        return self.board.has_three_in_a_row() != 0

    def get_winner(self):
        return self.board.has_three_in_a_row()

    def apply(self, player, move):
        if player == self.board.CROSS:
            self.board.set_cross(move)
        elif player == self.board.CIRCL:
            self.board.set_circle(move)
        else:
            raise NotImplementedError


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


if __name__ == '__main__':
    unittest.main()
