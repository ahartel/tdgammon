import random
import logging
from backgammon.board import Board
import copy


class Game:
    PLAYER1 = 0
    PLAYER2 = 1

    def __init__(self, board, do_log=False):
        self.board = board
        if do_log:
            self.logger = logging.getLogger('game_transcript')
            self.logger.setLevel(logging.INFO)
            fh = logging.FileHandler('../game.log')
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

    def move(self, evaluation_function):
        if self.last_dice[0] == self.last_dice[1]:
            final_moves = []
            my_fields_after = copy.copy(self.my_fields)
            other_fields_after = copy.copy(self.other_fields)
            for _ in range(4):
                moves = self.board.generate_possible_moves(self.last_dice[0], my_fields_after, other_fields_after)
                if len(moves) == 0:
                    return final_moves
                #print("Possible moves:\n{}".format(moves))
                used_die, best_move = evaluation_function(moves, my_fields_after, other_fields_after)
                #print("Best move: {}, {}".format(best_move, used_die))
                final_moves.append(best_move)
                try:
                    Game.apply_move(my_fields_after, other_fields_after, best_move)
                except Exception as e:
                    #print("dice {}, best_move {}".format(dice, best_move))
                    self.print_intermediate_board(my_fields_after, other_fields_after)
                    raise e
                #self.print_intermediate_board(my_fields_after, other_fields_after)

            return final_moves
        else:
            final_moves = []
            moves = []
            moves.extend(self.board.generate_possible_moves(self.last_dice[0], self.my_fields, self.other_fields))
            moves.extend(self.board.generate_possible_moves(self.last_dice[1], self.my_fields, self.other_fields))
            #print("Possible moves:\n{}".format(moves))
            if len(moves) == 0:
                return final_moves
            used_die, best_move = evaluation_function(moves, self.my_fields, self.other_fields)
            #print("Best move: {}, {}".format(best_move, used_die))
            final_moves.append(best_move)

            my_fields_after = copy.copy(self.my_fields)
            other_fields_after = copy.copy(self.other_fields)
            Game.apply_move(my_fields_after, other_fields_after, best_move)

            #self.print_intermediate_board(my_fields_after, other_fields_after)

            other_die = dice[0]
            if dice[0] == used_die:
                other_die = dice[1]
            moves = self.board.generate_possible_moves(other_die, my_fields_after, other_fields_after)
            #print("Possible moves:\n{}".format(moves))
            if len(moves) > 0:
                used_die, best_move = evaluation_function(moves, my_fields_after, other_fields_after)
                #print("Best move: {}, {}".format(best_move, used_die))
                final_moves.append(best_move)
            return final_moves
