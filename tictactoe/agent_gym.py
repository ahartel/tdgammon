from tictactoe.game import Game
from tictactoe.board import Board
from TDAgent import TD0Agent
from RandomAgent import RandomTTTAgent
import tqdm
import numpy as np
import matplotlib.pyplot as plt


def run_game_loop(board, agent1, agent2, do_print=False):
    game = Game(board)
    while not game.is_finished():
        dice = game.roll_dice()
        if do_print:
            board.print()
            print("Dice for AI1 were {}.".format(dice))
        ai_move = agent1.move(game)
        if do_print:
            print("Resulting move: {}.".format(ai_move))
        game.apply(ai_move)

        if game.is_finished():
            break

        agent1.learn(None)
        agent1.remember_board_state()

        dice = game.roll_dice()
        if do_print:
            board.print()
            print("Dice for AI2 were {}.".format(dice))
        ai_move = agent2.move(game)
        if do_print:
            print("Resulting move: {}.".format(ai_move))
        game.apply(ai_move)

        if game.is_finished():
            break

        agent2.learn(None)
        agent2.remember_board_state()

    winner = game.get_winner()
    if do_print:
        print("Winner is {}".format(Board.board_symbol(winner)))
    if winner == Game.PLAYER1:
        agent1.learn(np.array([1.0]))
        agent2.learn(np.array([0.0]))
        # agent1.backprop(np.array([1.0, 0.0]))
        # agent2.backprop(np.array([0.0, 1.0]))
    elif winner == Game.PLAYER2:
        agent1.learn(np.array([0.0]))
        agent2.learn(np.array([1.0]))
        # agent1.backprop(np.array([0.0, 1.0]))
        # agent2.backprop(np.array([1.0, 0.0]))
    else:
        pass

    return winner, game.get_num_moves()


def main():
    num_games = 40000
    episode_length = int(num_games / 100)
    do_print = False
    board = Board()

    # agent1 = TD0Agent(board, num_hidden=4, player=Game.PLAYER1)
    agent1 = RandomTTTAgent(board, num_hidden=4, player=Game.PLAYER1)

    agent2 = TD0Agent(board, num_hidden=4, player=Game.PLAYER2)
    # agent2 = RandomTTTAgent(board, num_hidden=4, player=Game.PLAYER2)

    # agent1.load_weights("agent1.final.weights")
    # agent2.load_weights("agent2.final.weights")

    winners = []

    for _ in tqdm.tqdm(range(num_games)):
        board.reinit()
        agent1.reset_trace()
        agent2.reset_trace()
        winner, num_moves = run_game_loop(board, agent1, agent2, do_print=do_print)
        winners.append(winner)

    # agent1.save_weights("agent1.final.weights")
    # agent2.save_weights("agent2.final.weights")

    print()
    print("---------------------------")
    print(" End of training reporting")
    print("---------------------------")

    print(" Overall results were:")
    results = np.bincount(winners, minlength=3)
    # Player 1 is +2, Player 2 is 0 and Draw is 1
    print("  Draw    : {:5d}, {:5.2f}".format(results[0], results[0] / num_games * 100.0))
    print("  Player 1: {:5d}, {:5.2f}".format(results[1], results[1] / num_games * 100.0))
    print("  Player 2: {:5d}, {:5.2f}".format(results[2], results[2] / num_games * 100.0))

    num_episodes = int(num_games / episode_length)
    fractions = np.zeros((num_episodes, 3))
    for episode in range(num_episodes):
        results_of_episode = winners[episode * episode_length:(episode + 1) * episode_length]
        fractions[episode] = np.array(np.bincount(results_of_episode, minlength=3), dtype='float') / episode_length

    if True:
        plt.plot(fractions[:, 0], label='Draw')
        plt.plot(fractions[:, 1], label='Player 1 (random)')
        plt.plot(fractions[:, 2], label='Player 2 (learn)')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
