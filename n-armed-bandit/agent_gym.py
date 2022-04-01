import torch

from MLP import PtRandomMLP
from n_armed_bandit import NBandit
from tqdm import tqdm
from torch import nn


def main():
    num_bandits = 10
    num_training_runs = 5000

    model = PtRandomMLP([num_bandits, 1])
    bandits = NBandit(num_bandits, avg=0.5, stddev=0.5)

    print("Initial comparison:")
    for bandit in range(num_bandits):
        inputs = torch.zeros(num_bandits)
        inputs[bandit] = 1.0
        print("  Bandit {}: {:2.3f} vs. {:2.3f}".format(bandit,
                                                        bandits.underlying_value(bandit),
                                                        model(inputs.float())[0]))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    print("Starting Training")
    model.train()
    for run in tqdm(range(num_training_runs)):
        for bandit in range(num_bandits):
            reward = torch.tensor([bandits.give_reward(bandit)])
            inputs = torch.zeros(num_bandits)
            inputs[bandit] = 1.0

            pred = model(inputs.float())
            loss = loss_fn(pred, reward)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("Finished training")

    print("Final comparison:")
    model.eval()
    for bandit in range(num_bandits):
        inputs = torch.zeros(num_bandits)
        inputs[bandit] = 1.0
        print("  Bandit {}: {:2.3f} vs. {:2.3f}".format(bandit,
                                                        bandits.underlying_value(bandit),
                                                        model(inputs)[0]))


if __name__ == '__main__':
    main()
