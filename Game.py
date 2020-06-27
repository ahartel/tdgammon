
class Game:
    PLAYER1 = 0
    PLAYER2 = 1

    def __init__(self, board):
        self.board = board

    def is_finished(self):
        white_won = self.board.whites[0] == 15
        black_won = self.board.blacks[0] == 15
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
            if fields_other[25-move[1]] <= 1:
                fields_mover[move[0]] -= 1
                fields_mover[move[1]] += 1
                if fields_other[25-move[1]] == 1:
                    fields_other[25-move[1]] = 0
                    fields_other[25] += 1
            else:
                raise Exception("Target field is blocked by opponent.")
        else:
            raise Exception("You don't have checkers here to move.")

    def apply(self, player, moves):
        for i, move in enumerate(moves):
            print("Applying move {} for player {}".format(i, player))
            try:
                if player == self.PLAYER1:
                    self.apply_move(self.board.whites, self.board.blacks, move)
                else:
                    self.apply_move(self.board.blacks, self.board.whites, move)
            except Exception as e:
                self.board.print()
                raise e

    @staticmethod
    def roll_dice():
        return 1, 1
