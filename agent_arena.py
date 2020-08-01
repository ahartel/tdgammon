import sys
from Game import Game
from TDAgent import TD0Agent
from RandomAgent import RandomAgent
from board import Board
import tqdm


def run_game_loop(board, agent1, agent2, do_print=False):
    game = Game(board)
    while not game.is_finished():
        dice = game.roll_dice()
        if do_print:
            board.print()
            print("Dice for AI1 were {}.".format(dice))
        ai_moves = agent1.move(dice)
        if do_print:
            print("Resulting moves: {}.".format(", ".join([str(move) for move in ai_moves])))
        game.apply(game.PLAYER1, ai_moves)

        if game.is_finished():
            break

        dice = game.roll_dice()
        if do_print:
            board.print()
            print("Dice for AI2 were {}.".format(dice))
        ai_moves = agent2.move(dice)
        if do_print:
            print("Resulting moves: {}.".format(", ".join([str(move) for move in ai_moves])))
        game.apply(game.PLAYER2, ai_moves)

    return game.get_winner()


def main():
    white_won = 0
    black_won = 0
    draws = 0
    num_games = 1000
    board = Board()
    agent1 = TD0Agent(board, use_whites=True)
    agent1.load_weights(sys.argv[1])
    agent2 = RandomAgent(board, use_whites=False)
    for i in tqdm.tqdm(range(num_games)):
        board.reinit()
        winner = run_game_loop(board, agent1, agent2)
        if winner < 0:
            black_won += 1
        elif winner > 0:
            white_won += 1
        else:
            draws += 1

    print("{:2f}% / {:2f}% white / black and {:2f}% draws".format(white_won/num_games*100.0,
                                                                  black_won/num_games*100.0,
                                                                  draws/num_games*100.0))


if __name__ == '__main__':
    main()
