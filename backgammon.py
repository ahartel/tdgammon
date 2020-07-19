from Game import Game
from RandomAgent import RandomAgent
from board import Board
import random
import re


def matches_any_dice_roll(fro, to, dice_roll):
    for roll in dice_roll:
        if fro - to == roll:
            return roll
    for roll in dice_roll:
        if (fro - to < roll) and to == 0:
            return roll
    return None


def get_move(dice_roll):
    move_regex = re.compile("\d{1,2}\/\d{1,2}")
    passed = False
    current_move = None
    while True:
        current_move = input("Please enter a move for {} steps:".format(",".join([str(roll) for roll in dice_roll])))
        if current_move == "":
            if passed is False:
                print("Empty move. Hit enter again to pass.")
                passed = True
                continue
            else:
                break
        passed = False
        match = move_regex.match(current_move)
        if not match:
            print("Move notation must contain two numbers separated by a slash, try again.")
            continue
        fro, to = current_move.split("/")
        fro = int(fro)
        to = int(to)
        if fro <= to:
            print("You're moving in the wrong direction, try again.")
            continue
        matching_roll = matches_any_dice_roll(fro, to, dice_roll)
        if matching_roll:
            dice_roll.remove(matching_roll)
            return fro, to
        else:
            print("Difference of starting and end position must match dice value {}, try again.".format(dice_roll))
            continue

    return ()


def present_dice_to_human_and_ask_move(dice_roll):
    print("Dice came out {}".format(dice_roll))
    moves = []
    dice = list(dice_roll)
    if dice_roll[0] == dice_roll[1]:
        dice.append(dice_roll[0])
        dice.append(dice_roll[0])
        for i in range(4):
            moves.append(get_move(dice))
    else:
        moves.append(get_move(dice))
        moves.append(get_move(dice))
    return moves


def main():
    #random.seed(42)
    board = Board()
    game = Game(board)
    agent = RandomAgent(board)
    while not game.is_finished():
        dice = game.roll_dice()
        board.print()
        allowed_moves_made = False
        while allowed_moves_made is False:
            human_moves = present_dice_to_human_and_ask_move(dice)
            try:
                allowed_moves_made = game.apply(game.PLAYER1, human_moves)
            except Exception as e:
                print(e)
                allowed_moves_made = False
        dice = game.roll_dice()
        ai_moves = agent.move(dice)
        print("Dice for AI were {}. Resulting moves: {}".format(dice, ", ".join([str(move) for move in ai_moves])))
        game.apply(game.PLAYER2, ai_moves)


if __name__ == '__main__':
    main()
