import unittest


class PositionAlreadyTaken(Exception):
    pass


class Board:
    CROSS = 1
    CIRCL = -1

    def __init__(self):
        self.positions = [[0 for _ in range(3)] for _ in range(3)]

    def __getitem__(self, position):
        return self.positions[position[0]][position[1]]

    def __setitem__(self, position, value):
        self.positions[position[0]][position[1]] = value

    def set_cross(self, position):
        self.assert_empty(position)
        self[position] = self.CROSS

    def assert_empty(self, position):
        pos_value = self[position]
        if pos_value != 0:
            raise PositionAlreadyTaken

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


class TTTBoardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board()

    def test_NewBoardIsEmpty(self):
        for row in range(3):
            for col in range(3):
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
