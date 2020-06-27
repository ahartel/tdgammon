from Game import Game
from RandomAgent import RandomAgent
from board import Board


def present_dice_to_human_and_ask_move(dice_roll):
    print("Dice came out {}".format(dice_roll))
    num_moves = 2
    if dice_roll[0] == dice_roll[1]:
        num_moves = 4
    moves = []
    i = 0
    while i < num_moves:
        current_move = input("Please enter a move {} of {}:".format(i+1, num_moves))
        if "/" not in current_move:
            print("Move notation must contain a slash, try again.")
            continue
        fro, to = current_move.split("/")
        fro = int(fro)
        to = int(to)
        if fro <= to:
            print("You're moving in the wrong direction, try again.")
            continue
        elif fro - to == dice_roll[0] or fro - to == dice_roll[1]:
            i += 1
            moves.append((fro, to))
        else:
            print("Difference of starting and end position must match a dice value, try again.")
            continue

    return moves


if __name__ == '__main__':
    board = Board()
    game = Game(board)
    agent = RandomAgent(board)
    while not game.is_finished():
        dice = game.roll_dice()
        board.print()
        human_moves = present_dice_to_human_and_ask_move(dice)
        game.apply(game.PLAYER1, human_moves)
        dice = game.roll_dice()
        ai_moves = agent.move(dice)
        game.apply(game.PLAYER2, ai_moves)
