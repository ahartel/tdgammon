from Game import Game
from RandomAgent import RandomAgent
from board import Board
import random


def run_game_loop(board, agent1, agent2):
    game = Game(board)
    while not game.is_finished():
        dice = game.roll_dice()
        board.print()
        print("Dice for AI1 were {}.".format(dice))
        ai_moves = agent1.move(dice)
        print("Resulting moves: {}.".format(", ".join([str(move) for move in ai_moves])))
        game.apply(game.PLAYER1, ai_moves)

        dice = game.roll_dice()
        board.print()
        print("Dice for AI2 were {}.".format(dice))
        ai_moves = agent2.move(dice)
        print("Resulting moves: {}.".format(", ".join([str(move) for move in ai_moves])))
        game.apply(game.PLAYER2, ai_moves)


def main():
    for _ in range(100):
        board = Board()
        agent1 = RandomAgent(board, use_whites=True)
        agent2 = RandomAgent(board, use_whites=False)
        run_game_loop(board, agent1, agent2)


if __name__ == '__main__':
    main()
