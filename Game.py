import random
import logging
from board import Board


class Game:
    PLAYER1 = 0
    PLAYER2 = 1

    def __init__(self, board, do_log=False):
        self.board = board
        if do_log:
            self.logger = logging.getLogger('game_transcript')
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler('game.log')
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        self.last_dice = None
        self.do_log = do_log
        self.__num_moves = 0

    def get_winner(self):
        white_won, black_won = self.get_winning_states()
        if white_won and black_won:
            return 0
        elif white_won and not black_won:
            return 1
        elif not white_won and black_won:
            return -1
        else:
            raise Exception("No one has won")

    def get_num_moves(self):
        return self.__num_moves

    def get_winning_states(self):
        white_won = self.board.whites[0] == 15
        black_won = self.board.blacks[0] == 15
        return white_won, black_won

    def is_finished(self):
        white_won, black_won = self.get_winning_states()
        return black_won or white_won

    @staticmethod
    def apply_move(fields_mover, fields_other, move):
        if fields_mover[25] > 0:
            if move[0] == 25:
                if fields_other[25-move[1]] <= 1:
                    fields_mover[25] -= 1
                    fields_mover[move[1]] += 1
                    if fields_other[25-move[1]] == 1:
                        fields_other[25-move[1]] -= 1
                        fields_other[25] += 1
                else:
                    raise Exception("Target field is blocked by opponent.")
            else:
                raise Exception("There are checkers on the bar that must be moved first.")
        elif fields_mover[move[0]] > 0:
            if move[1] == 0:
                if Board.all_checkers_in_home_quadrant(fields_mover):
                    fields_mover[0] += 1
                    fields_mover[move[0]] -= 1
                else:
                    raise Exception("All checkers must be in home quadrant to move checkers out.")
            elif fields_other[25-move[1]] <= 1:
                fields_mover[move[0]] -= 1
                fields_mover[move[1]] += 1
                if fields_other[25-move[1]] == 1:
                    fields_other[25-move[1]] = 0
                    fields_other[25] += 1
            else:
                raise Exception("Target field is blocked by opponent.")
        else:
            raise Exception("You don't have checkers here to move.")

    def log_moves(self, moves):
        self.logger.info("{}{}: {}".format(self.last_dice[0],
                                           self.last_dice[1],
                                           " ".join(["{}/{}".format(m0, m1) for m0, m1 in moves])))

    def apply(self, player, moves):
        if self.do_log:
            self.log_moves(moves)
        for i, move in enumerate(moves):
            if not bool(move):
                continue
            try:
                if player == self.PLAYER1:
                    self.apply_move(self.board.whites, self.board.blacks, move)
                else:
                    self.apply_move(self.board.blacks, self.board.whites, move)
            except Exception as e:
                self.board.print()
                print(moves)
                print(move)
                raise e

        self.__num_moves += 1

    def roll_dice(self):
        throw = random.randint(1, 6), random.randint(1, 6)
        self.last_dice = throw
        return throw
