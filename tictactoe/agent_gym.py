import random

from MLP import MLP
import numpy as np
from tictactoe.game import Game
from tictactoe.board import Board, Position, Move
from tqdm import tqdm
import matplotlib.pyplot as plt
from experienced_player import experienced_player


def random_player(board):
    possible_moves = []
    for row in range(Board.NUM_ROWS_AND_COLS):
        for col in range(Board.NUM_ROWS_AND_COLS):
            pos = Position(row, col)
            if board.is_empty(pos):
                possible_moves.append(Move(pos, Game.PLAYER2))
    if len(possible_moves) > 0:
        return random.choice(possible_moves)
    else:
        return None


def main():
    _lambda = None
    num_hidden_l1 = 50
    num_hidden_l2 = 10
    num_training_runs = 50000
    mlp = MLP(weights=[np.random.uniform(0.0, 0.1, (num_hidden_l1, 3 * Board.NUM_ROWS_AND_COLS * Board.NUM_ROWS_AND_COLS)),
                       np.random.uniform(0.0, 0.1, (num_hidden_l2, num_hidden_l1)),
                       np.random.uniform(0.0, 0.1, (1, num_hidden_l2))
                       ],
              biases=[np.zeros(num_hidden_l1),
                      np.zeros(num_hidden_l2),
                      np.zeros(1)],
              learning_rate=0.01)
    board = Board()
    game = Game(board)

    print("Initial run:")
    while not game.is_finished():
        possible_moves = []
        for row in range(Board.NUM_ROWS_AND_COLS):
            for col in range(Board.NUM_ROWS_AND_COLS):
                pos = Position(row, col)
                if board.is_empty(pos):
                    possible_moves.append(Move(pos, Game.PLAYER1))
        move = None
        if random.random() > 0.1:
            predicted_rewards = [0.0 for _ in possible_moves]
            for idx, move in enumerate(possible_moves):
                inputs = Board.get_network_inputs_of_board_state(board.get_copy_of_state())
                predicted_rewards[idx] = mlp.run_input(inputs, save_inputs_and_activations=False)
            max_idx = np.argmax(predicted_rewards)
            move = possible_moves[max_idx]
        else:
            move = random.choice(possible_moves)
        game.apply(move)
        # let opponent move
        opponent_move = random_player(board)
        # opponent_move = experienced_player(board)
        if opponent_move is not None and not game.is_finished():
            game.apply(opponent_move)
        board.print()
        print()
    print("Initial run done".format())

    print("Starting Training")
    path_lengths = np.zeros(num_training_runs)
    winners = np.zeros(num_training_runs)
    for run in tqdm(range(num_training_runs)):
        greedy_threshold = 0.1
        _lambda = 0.75

        # if run % batch_size == 0 and run > 0:
        #     batches = int(run/batch_size)
        #     mean_length
        #     = np.mean(path_lengths[(batches-1) * batch_size:batches * batch_size])
        #     print("Average path length of last {} runs: {}".format(batch_size, mean_length))
        #     fig, ax = plt.subplots()
        #     plottable_weights = mlp.get_weights()[0].reshape((width, height))
        #     im = ax.imshow(plottable_weights, cmap='Blues', interpolation='none')
        #     fig.colorbar(im)
        #     fig.savefig("weights_{}.png".format(batches))

        board.reinit()
        eligibility_traces = [np.zeros((num_hidden_l1, 3 * Board.NUM_ROWS_AND_COLS * Board.NUM_ROWS_AND_COLS)),
                              np.zeros((num_hidden_l2, num_hidden_l1)),
                              np.zeros((1, num_hidden_l2))]
        # eligibility_traces = [np.zeros((1, 2 * Board.NUM_ROWS_AND_COLS * Board.NUM_ROWS_AND_COLS))]
        last_pos_value = 0
        while not game.is_finished():
            next_pos = None
            # find new positions
            possible_moves = []
            for row in range(Board.NUM_ROWS_AND_COLS):
                for col in range(Board.NUM_ROWS_AND_COLS):
                    pos = Position(row, col)
                    if board.is_empty(pos):
                        possible_moves.append(Move(pos, Game.PLAYER1))
            move = None
            if random.random() > greedy_threshold:
                predicted_rewards = [0.0 for _ in possible_moves]
                for idx, move in enumerate(possible_moves):
                    inputs = Board.get_network_inputs_of_board_state(board.get_copy_of_state())
                    predicted_rewards[idx] = mlp.run_input(inputs, save_inputs_and_activations=False)
                max_idx = np.argmax(predicted_rewards)
                move = possible_moves[max_idx]
            else:
                move = random.choice(possible_moves)
            game.apply(move)
            # let opponent move
            opponent_move = random_player(board)
            # opponent_move = experienced_player(board)
            if opponent_move is not None and not game.is_finished():
                game.apply(opponent_move)

            # calculate the gradient based on last state
            current_gradient, biases = mlp.gradient(0)
            # run the input for the gradient of the next run
            # and for the value estimate
            inputs = Board.get_network_inputs_of_board_state(board.get_copy_of_state())
            this_pos_value = mlp.run_input(inputs, save_inputs_and_activations=True)
            eligibility_traces[0] = _lambda * eligibility_traces[0] + current_gradient[0]
            eligibility_traces[1] = _lambda * eligibility_traces[1] + current_gradient[1]
            eligibility_traces[2] = _lambda * eligibility_traces[2] + current_gradient[2]
            reward = 1.0 if game.get_winner() == Game.PLAYER1 else 0.0
            delta = reward + this_pos_value - last_pos_value
            mlp.add_to_weights([eligibility_traces[0] * delta,
                                eligibility_traces[1] * delta,
                                eligibility_traces[2] * delta
                                ])
            last_pos_value = this_pos_value
        winners[run] = game.get_winner()
    print("Finished training")

    draws = []
    player1_winner = []
    player2_winner = []
    average = []
    batch_size = 100
    for plot_run in range(int(num_training_runs/batch_size)):
        unique, counts = np.unique(winners[plot_run*batch_size:(plot_run+1)*batch_size], return_counts=True)
        # draws.append(counts[0])
        # player1_winner.append(counts[1])
        # player2_winner.append(counts[2])
        average.append(counts[1]/batch_size)

    fig, ax = plt.subplots()
    # ax.plot(draws, label="draws")
    # ax.plot(player1_winner, label="Player 1")
    # ax.plot(player2_winner, label="Player 2")
    ax.plot(average)
    ax.grid()
    ax.legend()
    fig.savefig("tictactoe/output/winners.png")

    print("Last batch statistics:")
    unique, counts = np.unique(winners[plot_run*batch_size:(plot_run + 1) * batch_size], return_counts=True)
    print("Player 1: {}%".format(counts[1]/batch_size*100.0))
    if len(counts) > 2:
        print("Player 2: {}%".format(counts[2]/batch_size*100.0))
    print("Draw:     {}%".format(counts[0]/batch_size*100.0))

    print("Final comparison:")
    board.reinit()
    while not game.is_finished():
        possible_moves = []
        for row in range(Board.NUM_ROWS_AND_COLS):
            for col in range(Board.NUM_ROWS_AND_COLS):
                pos = Position(row, col)
                if board.is_empty(pos):
                    possible_moves.append(Move(pos, Game.PLAYER1))
        move = None
        if random.random() > 0.1:
            predicted_rewards = [0.0 for _ in possible_moves]
            for idx, move in enumerate(possible_moves):
                inputs = Board.get_network_inputs_of_board_state(board.get_copy_of_state())
                predicted_rewards[idx] = mlp.run_input(inputs, save_inputs_and_activations=False)
            max_idx = np.argmax(predicted_rewards)
            move = possible_moves[max_idx]
        else:
            move = random.choice(possible_moves)
        game.apply(move)
        # let opponent move
        random_move = random_player(board)
        if random_move is not None and not game.is_finished():
            game.apply(random_move)
        board.print()
        print()
    print("Final run done".format())

    print("Final weights")
    # weights are plotted like this:
    # first row left to right, elements 0-4, second row left to right elements 5-9, and so on
    # thus, the first row shows the grid positions (0, 0), (1, 0), (2, 0) etc.
    # the second row shows positions (0, 1), (1, 1), (2, 1) etc.
    fig, ax = plt.subplots(2)
    for layer in range(2):
        plottable_weights = mlp.get_weights()[layer]
        im = ax[layer].imshow(plottable_weights, cmap='Blues', interpolation='none')
        plt.colorbar(im, ax=ax[layer])

    fig.savefig("tictactoe/output/weights_{}.png".format(int(num_training_runs)))

    # dummy_weights = np.arange(width * height)
    # plottable_weights = dummy_weights.reshape((width, height))
    # plt.imshow(plottable_weights, cmap='Blues', interpolation='none')
    # plt.colorbar()
    # plt.show()


if __name__ == '__main__':
    main()
