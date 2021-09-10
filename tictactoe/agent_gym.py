from tictactoe.game import Game
from tictactoe.board import Board
from TDAgent import TD0Agent
from RandomAgent import RandomAgent
import tqdm
import numpy as np


def run_game_loop(board, agent1, agent2, do_print=False):
    game = Game(board)
    while not game.is_finished():
        #agent1.learn()
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

        #agent2.learn()
        dice = game.roll_dice()
        if do_print:
            board.print()
            print("Dice for AI2 were {}.".format(dice))
        ai_move = agent2.move(game)
        if do_print:
            print("Resulting move: {}.".format(ai_move))
        game.apply(ai_move)

    winner = game.get_winner()
    if do_print:
        print("Winner is {}".format(Board.board_symbol(winner)))
    if winner > 0:
        agent1.learn(np.array([1.0, 0.0]))
        agent2.learn(np.array([0.0, 1.0]))
        #agent1.backprop(np.array([1.0, 0.0]))
        #agent2.backprop(np.array([0.0, 1.0]))
    else:
        agent1.learn(np.array([0.0, 1.0]))
        agent2.learn(np.array([1.0, 0.0]))
        #agent1.backprop(np.array([0.0, 1.0]))
        #agent2.backprop(np.array([1.0, 0.0]))

    return winner, game.get_num_moves()


def main():
    white_won = 0
    black_won = 0
    draws = 0
    num_games = 2
    board = Board()

    #agent1 = TD0Agent(board, player=Game.PLAYER1)
    #agent2 = TD0Agent(board, player=Game.PLAYER2)
    #agent1.load_weights("agent1.final.weights")
    #agent2.load_weights("agent2.final.weights")
    agent1 = RandomAgent(board, player=Game.PLAYER1)
    agent2 = RandomAgent(board, player=Game.PLAYER2)

    for i in tqdm.tqdm(range(num_games)):
        board.reinit()
        agent1.reset_trace()
        agent2.reset_trace()
        winner, num_moves = run_game_loop(board, agent1, agent2, do_print=True)
        if winner < 0:
            black_won += 1
            #print("Black won after {} moves".format(num_moves))
        elif winner > 0:
            white_won += 1
            #print("White won after {} moves".format(num_moves))
        else:
            draws += 1
            #print("Draw after {} moves".format(num_moves))

    agent1.save_weights("agent1.final.weights")
    agent2.save_weights("agent2.final.weights")
    print("{:2f}% / {:2f}% white / black and {:2f}% draws".format(white_won/num_games*100.0,
                                                                  black_won/num_games*100.0,
                                                                  draws/num_games*100.0))


if __name__ == '__main__':
    main()
