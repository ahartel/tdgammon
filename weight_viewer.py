import matplotlib.pyplot as plt
import numpy as np

def load_weights(self, filename):
    with open(filename, "rb") as file:
        hidden_weights = np.load(file)
        output_weights = np.load(file)
        hidden_biases = np.load(file)
        output_biases = np.load(file)
        return hidden_weights, output_weights, hidden_biases, output_biases


def plot_weights_and_diff(initial, final):
    plt.imshow(initial, interpolation='none')
    plt.savefig("initial.png")

    
def main(initial_weights_filename, final_weights_filename):
    hwi, owi, hbi, obi = load_weights(initial_weights_filename)
    hwf, owf, hbf, obf = load_weights(final_weights_filename)

    plot_weights_and_diff(hwi, hwf)

    
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

