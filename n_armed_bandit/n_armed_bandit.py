import numpy as np


class NBandit:
    def __init__(self, num_bandits, avg=0.0, stddev=1.0):
        self.num_bandits = num_bandits
        self.__bandits = np.random.normal(avg, stddev, num_bandits)

    def give_reward(self, n):
        if n >= self.num_bandits:
            raise IndexError
        return np.random.normal(self.__bandits[n], 1.0)

    def underlying_value(self, n):
        return self.__bandits[n]

    def __str__(self):
        return str(self.__bandits)


def main():
    num_bandits = 10
    num_runs = 200
    bandits = NBandit(num_bandits)
    results = np.zeros((num_bandits, num_runs))
    for bandit in range(num_bandits):
        for run in range(num_runs):
            results[bandit][run] = bandits.give_reward(bandit)
    print(np.mean(results, 1))
    print(bandits)


if __name__ == '__main__':
    main()
