from Game import Game
from RandomAgent import RandomAgent
from board import Board
import random


def main():
    board = Board()
    game = Game(board)
    agent1 = RandomAgent(board, use_whites=True)
    agent2 = RandomAgent(board, use_whites=False)
    while not game.is_finished():
        dice = game.roll_dice()
        board.print()
        ai_moves = agent1.move(dice)
        print("Dice for AI1 were {}. Resulting moves: {}".format(dice, ", ".join([str(move) for move in ai_moves])))
        game.apply(game.PLAYER1, ai_moves)

        dice = game.roll_dice()
        board.print()
        ai_moves = agent2.move(dice)
        print("Dice for AI2 were {}. Resulting moves: {}".format(dice, ", ".join([str(move) for move in ai_moves])))
        game.apply(game.PLAYER2, ai_moves)


if __name__ == '__main__':
    main()
