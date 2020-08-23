import sys
import matplotlib.pyplot as plt
import numpy as np


def load_weights(filename):
    with open(filename, "rb") as file:
        hidden_weights = np.load(file)
        output_weights = np.load(file)
        hidden_biases = np.load(file)
        output_biases = np.load(file)
        return hidden_weights, output_weights, hidden_biases, output_biases


def plot_weights_and_diff(initial, final):
    fig, ax = plt.subplots(3, 1)
    diff = final-initial

    im = ax[0].pcolormesh(initial)
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title('initial')

    im = ax[1].pcolormesh(final)
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title('final')

    im = ax[2].pcolormesh(diff)
    ax[2].set_title('diff')
    plt.colorbar(im, ax=ax[2])

    plt.show()

    
def main(initial_weights_filename, final_weights_filename):
    hwi, owi, hbi, obi = load_weights(initial_weights_filename)
    hwf, owf, hbf, obf = load_weights(final_weights_filename)

    plot_weights_and_diff(hwi, hwf)
    plot_weights_and_diff(owi, owf)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

