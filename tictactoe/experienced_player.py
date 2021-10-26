import numpy as np
from tictactoe.board import Board, Move, Position
from tictactoe.game import Game


BOARD_SIZE = 3
BLOCKING_EXPERT = False


def number_of_matches(match):
    return len(match[0])


def map_pseudo_to_normal(pseudo_row, pseudo_col):
    """Some functions use a pseud-row coordinate system. This system consists
    of rows and columns but the columns are defined in the sense of the
    function extract_all_rows. Columns are then the index within that row. This
    function maps those coordinates back to the normal numpy ones."""
    # the first BS pseudo rows are actual rows
    if pseudo_row < BOARD_SIZE:
        return pseudo_row, pseudo_col
    # the second set of BS rows are actual columns
    elif BOARD_SIZE <= pseudo_row < 2 * BOARD_SIZE:
        # here, we just need to transpose them
        return pseudo_col, (pseudo_row - BOARD_SIZE)
    # the last to indices stand for diagonal and anti-diagonal, respectively
    elif pseudo_row == 2*BOARD_SIZE:
        # the default diagonal
        idx = 1
        return idx * pseudo_col, idx * pseudo_col
    elif pseudo_row == 2*BOARD_SIZE+1:
        # the anti-diagonal, x increases, y decreases
        idx = 0
        idy = BOARD_SIZE - 1
        return idy - pseudo_col, idx + pseudo_col


def extract_all_rows(board):
    """Generator function for all rows, columns and diagonals of the board"""
    # return the rows first
    for row in range(BOARD_SIZE):
        yield board[row, :]
    # then the columns
    for col in range(BOARD_SIZE):
        yield board[:, col]
    # then the diagonal
    yield np.diagonal(board)
    # and last the anti-diagonal
    yield np.diagonal(np.flipud(board))


def experienced_player(board):
    my_mark = Game.PLAYER2
    op_mark = Game.PLAYER1
    board = board.get_copy_of_state()
    """Takes two marks (symbol value) for self and opponent and a board and returns
    a board with an additional mark."""
    match = np.where(board > 0)
    # This if statement implements a case on match
    if number_of_matches(match) == 0:
        # empty board, choose center
        return Move(Position(1, 1), my_mark)
    elif number_of_matches(match) == 1:
        # first move taken as center:
        if board[1, 1] == 0:
            return Move(Position(1, 1), my_mark)
        else:
            for (x, y) in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                if board[y, x] == 0:
                    return Move(Position(y, x), my_mark)
    else:
        # if there are two marks or more, iterate over all rows, cols, diagonals
        # store rows that can be completed and rows that need to be blocked
        completable = []
        blockable = []
        for (num, values) in enumerate(extract_all_rows(board)):
            # if the row is complete, take the next one
            number_of_marks = number_of_matches(np.where(values > 0))
            if number_of_marks == 3:
                continue
            # if the row can be completed by this player, then do so
            my_marks = np.where(values == my_mark)
            # print my_marks
            if number_of_matches(my_marks) == 2:
                completable.append((num, my_marks))
            op_marks = np.where(values == op_mark)
            # print op_marks
            if number_of_matches(op_marks) == 2:
                blockable.append((num, op_marks))

        verbs = ["completing"]
        lists = [completable]
        if BLOCKING_EXPERT:
            verbs.append("blocking")
            lists.append(blockable)

        for verb, lst in zip(verbs, lists):
            if len(lst) > 0:
                free_space = lst[0]
                # print free_space
                for col in range(BOARD_SIZE):
                    if col in free_space[1][0]:
                        continue
                    else:
                        # print verb,"field",free_space[0], col
                        (y, x) = map_pseudo_to_normal(free_space[0], col)
                        return Move(Position(y, x), my_mark)

        # last resort: randomness
        # print "Random"
        empty_fields = np.where(board == 0)
        # print(empty_fields)
        field = np.random.randint(len(empty_fields[0]))
        return Move(Position(empty_fields[0][field], empty_fields[1][field]), my_mark)
