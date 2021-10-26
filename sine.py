import matplotlib.pyplot as plt
import numpy
import tqdm

from MLP import MLP, Identity, Sigmoid
import numpy as np
import random


def plot_sine(mlp):
    xs = np.arange(0, 2*numpy.pi, 0.1)
    plt.plot(xs, np.sin(xs))
    outputs = []
    for x in xs:
        output = mlp.run_input([x])
        outputs.append(output[0] - output[1])
    plt.plot(xs, outputs)
    plt.show()


def main():
    num_inputs = 1
    num_outputs = 2
    num_hidden = 20
    mlp = MLP \
            (weights=[
            np.random.uniform(0.0, 0.1, (num_hidden, num_inputs)),
            np.random.uniform(0.0, 0.1, (num_outputs, num_hidden))
        ],
            biases=[np.zeros(num_hidden),
                    np.zeros(num_outputs)],
            learning_rate=0.01,
            activation_function=Sigmoid
        )

    plot_sine(mlp)

    num_runs = 100000
    for run in tqdm.tqdm(range(num_runs)):
        x = random.uniform(0, 2 * numpy.pi)
        output = mlp.run_input([x], save_inputs_and_activations=True)
        sine = np.sin(x)
        expected = [sine if sine > 0 else 0.0,
                    -sine if sine < 0 else 0.0]
        # mlp.backprop(expected)

        weight_updates = [None, None]
        bias_updates = [None, None]
        for idx, exp in enumerate(expected):
            weight_updates[idx], bias_updates[idx] = mlp.gradient(idx)
        for idx, exp in enumerate(expected):
            error = exp - output[idx]
            mlp.add_to_weights([weight_updates[idx][0] * error,
                                weight_updates[idx][1] * error])
            mlp.add_to_biases([bias_updates[idx][0] * error,
                               bias_updates[idx][1] * error])

    plot_sine(mlp)


if __name__ == '__main__':
    main()
