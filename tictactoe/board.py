import copy
import unittest
import itertools


class PositionAlreadyTaken(Exception):
    def __init__(self, pos, positions):
        self.message = "{}, {}".format(pos, positions)
        super().__init__(self.message)


class Position:

    def __init__(self, row, col):
        self.__row = row
        self.__col = col

    def __getitem__(self, item):
        if item == 0:
            return self.__row
        elif item == 1:
            return self.__col
        else:
            raise NotImplementedError

    def __str__(self):
        return "(r{}, c{})".format(self.__row, self.__col)


class Move:

    def __init__(self, position, player):
        self.__pos = position
        self.__player = player

    def get_pos(self):
        return self.__pos

    def get_player(self):
        return self.__player

    def __str__(self):
        return str(self.__pos) + "->" + Board.board_symbol(self.__player)


class Board:
    CROSS = 1
    CIRCL = -1
    NUM_ROWS_AND_COLS = 3

    def __init__(self):
        self.positions = None
        self.reinit()

    def reinit(self):
        self.positions = [[0 for _ in range(self.NUM_ROWS_AND_COLS)] for _ in range(self.NUM_ROWS_AND_COLS)]

    def __getitem__(self, position):
        return self.positions[position[0]][position[1]]

    def __setitem__(self, position, value):
        self.positions[position[0]][position[1]] = value

    def set_cross(self, position):
        self.assert_empty(position)
        self[position] = self.CROSS

    def all_positions_full(self):
        return all(a != 0 and b != 0 and c != 0 for (a, b, c) in self.positions)

    def is_empty(self, position):
        return self[position] == 0

    def assert_empty(self, position):
        if not self.is_empty(position):
            raise PositionAlreadyTaken(position, self.positions)

    def set_circle(self, position):
        self.assert_empty(position)
        self[position] = self.CIRCL

    def has_three_in_a_row(self):
        for row in range(3):
            if self[row, 0] == self[row, 1] and self[row, 1] == self[row, 2] and self[row, 0] != 0:
                return self[row, 0]
        for col in range(3):
            if self[0, col] == self[1, col] and self[1, col] == self[2, col] and self[0, col] != 0:
                return self[0, col]
        if self[0, 0] == self[1, 1] and self[1, 1] == self[2, 2] and self[0, 0] != 0:
            return self[0, 0]
        elif self[2, 0] == self[1, 1] and self[1, 1] == self[0, 2] and self[1, 1] != 0:
            return self[1, 1]
        else:
            return 0

    def get_copy_of_state(self):
        return copy.copy(self.positions)

    def get_network_inputs_with_move_applied(self, move):
        state = copy.deepcopy(self.positions)
        pos = move.get_pos()
        self.assert_empty(pos)
        state[pos[0]][pos[1]] = move.get_player()
        return list(itertools.chain.from_iterable(state))

    def get_network_input_size(self):
        return self.NUM_ROWS_AND_COLS * self.NUM_ROWS_AND_COLS

    @staticmethod
    def board_symbol(value):
        if value == 0:
            return "."
        else:
            return "X" if value > 0 else "O"

    def print(self):
        for row in range(self.NUM_ROWS_AND_COLS):
            print("|".join([self.board_symbol(x) for x in self.positions[row]]))
            if row < self.NUM_ROWS_AND_COLS-1:
                print("-+-+-")


class TTTBoardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board()

    def test_NewBoardIsEmpty(self):
        for row in range(Board.NUM_ROWS_AND_COLS):
            for col in range(Board.NUM_ROWS_AND_COLS):
                self.assertEqual(self.board[row, col], 0)

    def test_BoardCanSetCrosses(self):
        row = col = 1
        self.board.set_cross((row, col))
        self.assertEqual(self.board[row, col], 1)

    def test_BoardCanSetCircles(self):
        row = col = 1
        self.board.set_circle((row, col))
        self.assertEqual(self.board[row, col], -1)

    def test_BoardRaisesIfPositionTaken(self):
        row = col = 1
        self.board.set_circle((row, col))
        with self.assertRaises(PositionAlreadyTaken):
            self.board.set_cross((row, col))


if __name__ == '__main__':
    unittest.main()
