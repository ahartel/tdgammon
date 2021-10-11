import random

from MLP import MLP
import numpy as np
from grid import Grid
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    width = 5
    height = 5
    _lambda = None
    num_training_runs = 20000
    mlp = MLP(weights=[np.random.uniform(0.1, 0.05, (1, width * height))],
              biases=[np.zeros(1)],
              learning_rate=0.01)
    grid = Grid(width, height)

    print("Initial run:")
    pos = grid.get_start_position()
    naive_path = []
    while pos != grid.get_goal_position():
        new_positions = [
            (pos[0] + 0, pos[1] + 1),
            (pos[0] + 0, pos[1] - 1),
            (pos[0] - 1, pos[1] + 0),
            (pos[0] + 1, pos[1] + 0),
        ]
        if random.random() > 0.1:
            predicted_rewards = [0.0 for _ in range(4)]
            for idx, new_pos in enumerate(new_positions):
                if 0 <= new_pos[0] < width and 0 <= new_pos[1] < height:
                    inputs = np.zeros(width * height)
                    inputs[new_pos[0]+width*new_pos[1]] = 1.0
                    predicted_rewards[idx] = mlp.run_input(inputs)
            max_idx = np.argmax(predicted_rewards)
            pos = new_positions[max_idx]
        else:
            pos = (-1, -1)
            while not (0 <= pos[0] < width and 0 <= pos[1] < height):
                pos = random.choice(new_positions)
        assert(0 <= pos[0] < width and 0 <= pos[1] < height)
        naive_path.append(pos)
    print("Naive path took {} steps".format(len(naive_path)))

    print("Starting Training")
    batch_size = 2000
    path_lengths = np.zeros(num_training_runs)
    for run in tqdm(range(num_training_runs)):
        greedy_threshold = 0.5
        _lambda = 0.5

        if run % batch_size == 0 and run > 0:
            batches = int(run/batch_size)
            mean_length = np.mean(path_lengths[(batches-1) * batch_size:batches * batch_size])
            print("Average path length of last {} runs: {}".format(batch_size, mean_length))
            fig, ax = plt.subplots()
            plottable_weights = mlp.get_weights()[0].reshape((width, height))
            im = ax.imshow(plottable_weights, cmap='Blues', interpolation='none')
            fig.colorbar(im)
            fig.savefig("weights_{}.png".format(batches))
        pos = grid.get_start_position()
        path_length = 0
        eligibility_trace = np.zeros((1, width * height))
        last_pos_value = 0
        while pos != grid.get_goal_position():
            next_pos = None
            # find new positions
            new_positions = [
                (pos[0] + 0, pos[1] + 1),
                (pos[0] + 0, pos[1] - 1),
                (pos[0] - 1, pos[1] + 0),
                (pos[0] + 1, pos[1] + 0),
            ]
            # choose a position either greedily
            do_greedy_move = random.random() > greedy_threshold
            if do_greedy_move:
                predicted_rewards = [0.0 for _ in range(4)]
                for idx, new_pos in enumerate(new_positions):
                    if 0 <= new_pos[0] < width and 0 <= new_pos[1] < height:
                        inputs = np.zeros(width * height)
                        inputs[new_pos[0] + width * new_pos[1]] = 1.0
                        predicted_rewards[idx] = mlp.run_input(inputs)
                max_idx = np.argmax(predicted_rewards)
                next_pos = new_positions[max_idx]
            # or randomly
            else:
                next_pos = (-1, -1)
                while not (0 <= next_pos[0] < width and 0 <= next_pos[1] < height):
                    next_pos = random.choice(new_positions)
            assert (0 <= next_pos[0] < width and 0 <= next_pos[1] < height)
            path_length += 1
            # calculate the gradient based on last state
            current_gradient = mlp.gradient()[0][0]
            # run the input for the gradient of the next run
            # and for the value estimate
            inputs = np.zeros(width * height)
            inputs[next_pos[0] + width * next_pos[1]] = 1.0
            this_pos_value = mlp.run_input(inputs, save_inputs_and_activations=True)
            if do_greedy_move:
                eligibility_trace = _lambda * eligibility_trace + current_gradient
                delta = grid.give_reward(pos) + this_pos_value - last_pos_value
                mlp.add_to_weights([eligibility_trace * delta])
            last_pos_value = this_pos_value
            pos = next_pos
        path_lengths[run] = path_length
    print("Finished training")

    plot_values = []
    for plot_run in range(int(num_training_runs/100)):
        plot_values.append(np.mean(path_lengths[plot_run:(plot_run+1)*100]))
    fig, ax = plt.subplots()
    ax.plot(plot_values)
    ax.grid()
    fig.savefig("path_lengths.png")

    print("Final comparison:")
    pos = grid.get_start_position()
    naive_path = []
    while pos != grid.get_goal_position():
        new_positions = [
            (pos[0] + 0, pos[1] + 1),
            (pos[0] + 0, pos[1] - 1),
            (pos[0] - 1, pos[1] + 0),
            (pos[0] + 1, pos[1] + 0),
        ]
        if random.random() > 0.1:
            predicted_rewards = [0.0 for _ in range(4)]
            for idx, new_pos in enumerate(new_positions):
                if 0 <= new_pos[0] < width and 0 <= new_pos[1] < height:
                    inputs = np.zeros(width * height)
                    inputs[new_pos[0] + width * new_pos[1]] = 1.0
                    predicted_rewards[idx] = mlp.run_input(inputs)
            max_idx = np.argmax(predicted_rewards)
            pos = new_positions[max_idx]
        else:
            pos = (-1, -1)
            while not (0 <= pos[0] < width and 0 <= pos[1] < height):
                pos = random.choice(new_positions)
        assert (0 <= pos[0] < width and 0 <= pos[1] < height)
        naive_path.append(pos)
    print("Final path took {} steps".format(len(naive_path)))

    print("Final weights")
    # weights are plotted like this:
    # first row left to right, elements 0-4, second row left to right elements 5-9, and so on
    # thus, the first row shows the grid positions (0, 0), (1, 0), (2, 0) etc.
    # the second row shows positions (0, 1), (1, 1), (2, 1) etc.
    fig, ax = plt.subplots()
    plottable_weights = mlp.get_weights()[0].reshape((width, height))
    im = ax.imshow(plottable_weights, cmap='Blues', interpolation='none')
    fig.colorbar(im)
    fig.savefig("weights_{}.png".format(int(num_training_runs/batch_size)))

    # dummy_weights = np.arange(width * height)
    # plottable_weights = dummy_weights.reshape((width, height))
    # plt.imshow(plottable_weights, cmap='Blues', interpolation='none')
    # plt.colorbar()
    # plt.show()


if __name__ == '__main__':
    main()
