from MLP import MLP
import numpy as np
from n_armed_bandit import NBandit
from tqdm import tqdm


def main():
    num_bandits = 10
    num_training_runs = 1000
    mlp = MLP(weights=[np.random.uniform(0.0, 0.1, (1, num_bandits))],
              biases=[np.zeros(1)])
    bandits = NBandit(num_bandits, avg=0.5, stddev=0.5)

    print("Initial comparison:")
    for bandit in range(num_bandits):
        inputs = np.zeros(num_bandits)
        inputs[bandit] = 1.0
        print("  Bandit {}: {:2.3f} vs. {:2.3f}".format(bandit,
                                                        bandits.underlying_value(bandit),
                                                        mlp.run_input(inputs)[0]))

    print("Starting Training")
    for run in tqdm(range(num_training_runs)):
        for bandit in range(num_bandits):
            reward = bandits.give_reward(bandit)
            inputs = np.zeros(num_bandits)
            inputs[bandit] = 1.0
            mlp.run_input(inputs, save_inputs_and_activations=True)
            mlp.backprop(reward)
    print("Finished training")

    print("Final comparison:")
    for bandit in range(num_bandits):
        inputs = np.zeros(num_bandits)
        inputs[bandit] = 1.0
        print("  Bandit {}: {:2.3f} vs. {:2.3f}".format(bandit,
                                                        bandits.underlying_value(bandit),
                                                        mlp.run_input(inputs)[0]))


if __name__ == '__main__':
    main()
