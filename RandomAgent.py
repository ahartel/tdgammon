import MLP
from Game import Game
import copy
from board import Board


class RandomAgent:
    def __init__(self, board, use_whites=False):
        if use_whites:
            self.my_fields = board.whites
            self.other_fields = board.blacks
        else:
            self.my_fields = board.blacks
            self.other_fields = board.whites
        self.__use_whites = use_whites
        self.mlp = MLP.RandomThreeLayerMLP((board.NUM_FIELDS * 2 * 4) + 2 + 2, 40, 2)
        self.last_my_fields_before_move = None
        self.last_other_fields_before_move = None

    def evaluate_moves_by_mlp(self, moves, my_fields, other_fields):
        max_move_index = 0
        max_move_value = 0
        for idx, move_die in enumerate(moves):
            die, move = move_die
            my_fields_after = copy.copy(my_fields)
            other_fields_after = copy.copy(other_fields)
            try:
                Game.apply_move(my_fields_after, other_fields_after, move)
                inputs = Board.prepare_any_inputs(my_fields_after, other_fields_after)
                outputs = self.mlp.run_input(inputs)
                if outputs[0] > max_move_value:
                    max_move_value = max_move_value
                    max_move_index = idx
            except Exception as e:
                continue

        return moves[max_move_index]

    def print_intermediate_board(self, my_fields, other_fields):
        print("Intermediate board:")
        if self.__use_whites:
            Board(my_fields, other_fields).print()
        else:
            Board(other_fields, my_fields).print()

    def move(self, dice):
        self.last_my_fields_before_move = copy.copy(self.my_fields)
        self.last_other_fields_before_move = copy.copy(self.other_fields)
        if dice[0] == dice[1]:
            final_moves = []
            my_fields_after = copy.copy(self.my_fields)
            other_fields_after = copy.copy(self.other_fields)
            for _ in range(4):
                moves = self.generate_possible_moves(dice[0], my_fields_after, other_fields_after)
                if len(moves) == 0:
                    return final_moves
                #print("Possible moves:\n{}".format(moves))
                used_die, best_move = self.evaluate_moves_by_mlp(moves, my_fields_after, other_fields_after)
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
            moves.extend(self.generate_possible_moves(dice[0], self.my_fields, self.other_fields))
            moves.extend(self.generate_possible_moves(dice[1], self.my_fields, self.other_fields))
            #print("Possible moves:\n{}".format(moves))
            if len(moves) == 0:
                return final_moves
            used_die, best_move = self.evaluate_moves_by_mlp(moves, self.my_fields, self.other_fields)
            #print("Best move: {}, {}".format(best_move, used_die))
            final_moves.append(best_move)

            my_fields_after = copy.copy(self.my_fields)
            other_fields_after = copy.copy(self.other_fields)
            Game.apply_move(my_fields_after, other_fields_after, best_move)

            #self.print_intermediate_board(my_fields_after, other_fields_after)

            other_die = dice[0]
            if dice[0] == used_die:
                other_die = dice[1]
            moves = self.generate_possible_moves(other_die, my_fields_after, other_fields_after)
            #print("Possible moves:\n{}".format(moves))
            if len(moves) > 0:
                used_die, best_move = self.evaluate_moves_by_mlp(moves, my_fields_after, other_fields_after)
                #print("Best move: {}, {}".format(best_move, used_die))
                final_moves.append(best_move)
            return final_moves

    @staticmethod
    def generate_possible_moves(die, my_fields, other_fields):
        #print("Other fields in generate_possible_moves: {}".format(other_fields))
        moves = []
        if my_fields[25] > 0:
            if other_fields[die] <= 1:
                moves.append((die, [25, 25-die]))
                return moves
            else:
                return []
        for idx in range(1, 25):
            if my_fields[idx] > 0:
                if Board.all_checkers_in_home_quadrant(my_fields):
                    if idx-die <= 0:
                        moves.append((die, [idx, 0]))
                    elif idx-die > 0 and other_fields[25-(idx-die)] <= 1:
                        moves.append((die, [idx, idx-die]))
                else:
                    if idx-die > 0 and other_fields[25-(idx-die)] <= 1:
                        moves.append((die, [idx, idx-die]))
        return moves
