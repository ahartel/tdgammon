import numpy as np
import unittest


class ThreeLayerMLP:

    def __init__(self, num_inputs, num_hiddens, num_outputs, init_random=True, log=False):
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs
        if init_random:
            self.input_weights = np.random.standard_normal((num_hiddens, num_inputs))
            self.output_weights = np.random.standard_normal((num_outputs, num_hiddens))
        else:
            self.input_weights = np.zeros((num_hiddens, num_inputs))
            self.output_weights = np.zeros((num_outputs, num_hiddens))
        self.log("Initial input weights {}".format(self.input_weights))
        self.log("Initial output weights {}".format(self.output_weights))
        self.last_activations = [np.zeros((num_hiddens,)),
                                 np.zeros((num_outputs,))]
        self.last_inputs = np.zeros((num_inputs,))
        self.__lambda = 0.2
        self.__do_log = log

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(s):
        return s * (1 - s)

    def run_input(self, inputs):
        hidden = self.sigmoid(self.input_weights.dot(inputs))
        outputs = self.sigmoid(self.output_weights.dot(hidden))
        self.last_activations[0] = hidden
        self.last_activations[1] = outputs
        self.last_inputs = inputs
        return outputs

    def log(self, str):
        if self.__do_log:
            print(str)

    def backprop(self, expected):
        error = expected - self.last_activations[1]
        self.log("Error {}".format(error))
        last_level_derivative = self.sigmoid_derivative(self.last_activations[1])
        self.log("Error*deriv. {}".format(last_level_derivative))
        self.log("Last hidden activations {}".format(self.last_activations[0]))
        delta_w_output = np.outer(error * last_level_derivative, self.last_activations[0])
        self.log("Delta W {}".format(delta_w_output))

        hidden_errors = np.zeros((self.num_hiddens,))
        derivatives = self.sigmoid_derivative(self.last_activations[0])
        self.log("Derivatives {}".format(derivatives))
        for k in range(self.num_hiddens):
            for j in range(self.num_outputs):
                hidden_errors[k] += error[j] * self.output_weights[j, k] * derivatives[k]
        self.log("Hidden errors {}".format(hidden_errors))
        self.log("Last inputs {}".format(self.last_inputs))
        delta_w = np.outer(hidden_errors, self.last_inputs)
        self.log("Delta W {}".format(delta_w))

        self.output_weights = self.output_weights + self.__lambda * delta_w_output
        self.log("New output weights {}".format(self.output_weights))

        self.input_weights = self.input_weights + self.__lambda * delta_w
        self.log("New input weights {}".format(self.input_weights))


class ThreeLayerMLPTest(unittest.TestCase):
    @unittest.skip
    @staticmethod
    def test_XOR():
        mlp = ThreeLayerMLP(2, 2, 1, init_random=True)

        for i in range(1000):
            # print("Output {}".format(mlp.run_input(np.array([1.0, 0.0]))))
            mlp.backprop(np.ones((1,)))
            # print("Output {}".format(mlp.run_input(np.array([0.0, 1.0]))))
            mlp.backprop(np.ones((1,)))
            # print("Output {}".format(mlp.run_input(np.array([0.0, 0.0]))))
            mlp.backprop(np.zeros((1,)))
            # print("Output {}".format(mlp.run_input(np.array([1.0, 1.0]))))
            mlp.backprop(np.zeros((1,)))


class MLP:
    def __init__(self, weights, biases, learning_rate=0.1):
        assert (len(weights) == len(biases))
        self.__num_layers = len(weights)
        self.__weights = weights
        self.__biases = biases
        # for layer in weights:
        #    print(layer.shape)
        self.last_activations = [None for _ in range(self.__num_layers)]
        self.last_input = None
        self.learning_rate = learning_rate

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(s):
        return s * (1 - s)

    def run_input(self, input):
        self.last_input = input
        self.last_activations[0] = self.sigmoid(self.__weights[0].dot(input) + self.__biases[0])
        for layer in range(1, self.__num_layers):
            u = self.__weights[layer].dot(self.last_activations[layer - 1]) + self.__biases[layer]
            self.last_activations[layer] = self.sigmoid(u)
        return self.last_activations[self.__num_layers - 1]

    def backprop(self, expected):
        for layer in range(self.__num_layers, 0, -1):
            # calculate error signal
            if layer > 1:
                next_layer_output = self.last_activations[layer - 2]
            else:
                next_layer_output = self.last_input
            this_layer_output = self.last_activations[layer - 1]
            if layer == self.__num_layers:
                error_signal = (expected - this_layer_output) * self.sigmoid_derivative(this_layer_output)
            else:
                shape = self.__weights[layer].shape
                error_signal = np.zeros(shape[1])
                derivative = self.sigmoid_derivative(this_layer_output)
                for k in range(shape[1]):
                    for j in range(shape[0]):
                        error_signal[k] += self.__weights[layer][j, k] * previous_level_error[j] * derivative[k]
            previous_level_error = error_signal
            # calculate gradient
            delta_w = np.outer(error_signal, next_layer_output)
            # update
            self.__weights[layer - 1] += self.learning_rate * delta_w
            self.__biases[layer - 1] += self.learning_rate * error_signal

    def get_weights(self):
        return self.__weights

    def get_biases(self):
        return self.__biases


class MLPTest(unittest.TestCase):
    def test_InputOutput(self):
        mlp = MLP([np.zeros((1,))], [np.zeros((1,))])
        output = mlp.run_input(np.ones((1)))[0]
        self.assertAlmostEqual(output, 0.5, 3)

        output = mlp.run_input(np.zeros((1)))[0]
        self.assertAlmostEqual(output, 0.5, 3)

        mlp = MLP([np.ones((1,))], [np.zeros((1,))])
        output = mlp.run_input(np.ones((1)))[0]
        self.assertAlmostEqual(output, 0.731, 3)

        mlp = MLP([np.ones((2,))], [np.zeros((2,))])
        output = mlp.run_input(np.ones((2)))[0]
        self.assertGreater(output, 0.731)

    def test_sigmoid(self):
        self.assertAlmostEqual(MLP.sigmoid(2.0), 0.8807, 3)
        self.assertAlmostEqual(MLP.sigmoid(1.0), 0.731, 3)
        self.assertAlmostEqual(MLP.sigmoid(0.0), 0.5, 3)
        self.assertAlmostEqual(MLP.sigmoid(-1.0), 0.2689, 3)
        self.assertAlmostEqual(MLP.sigmoid(-2.0), 0.1192, 3)

    def test_XOR_feedforward(self):
        hidden_weights = np.ones((2, 2))
        hidden_weights[0, 0] = 20.0
        hidden_weights[0, 1] = 20.0
        hidden_weights[1, 0] = -20.0
        hidden_weights[1, 1] = -20.0
        output_weights = np.ones((1, 2)) * 20.0
        biases = [np.ones(2), np.ones(1)]
        biases[0][0] = -10.0
        biases[0][1] = 30.0
        biases[1][0] = -30.0
        mlp = MLP([hidden_weights, output_weights], biases)
        input0 = np.zeros(2)
        input1 = np.ones(2)
        input2 = np.zeros(2)
        input2[0] = 1.0
        input3 = np.zeros(2)
        input3[1] = 1.0
        output = mlp.run_input(input0)[0]
        self.assertLess(output, 0.01)
        output = mlp.run_input(input1)[0]
        self.assertLess(output, 0.01)
        output = mlp.run_input(input2)[0]
        self.assertGreater(output, 0.99)
        output = mlp.run_input(input3)[0]
        self.assertGreater(output, 0.99)

    def test_basic_backprop(self):
        mlp = MLP([np.zeros((1, 1))], [np.zeros((1, 1))])
        mlp.run_input(np.ones(1))[0]
        mlp.backprop(np.ones((1,)) * 0.8)
        weights = mlp.get_weights()
        self.assertGreater(weights[0][0, 0], 0.0)
        biases = mlp.get_biases()
        self.assertGreater(biases[0][0, 0], 0.0)

    def test_first_layer_updates_even_with_zero_weights(self):
        mlp = MLP(weights=[np.zeros((1, 1)), np.ones((1, 1))],
                  biases=[np.zeros(1), np.zeros(1)])
        mlp.run_input(np.ones(1))
        mlp.backprop(np.ones(1) * 0.8)
        weights = mlp.get_weights()
        self.assertGreater(weights[0][0, 0], 0.0)
        biases = mlp.get_biases()
        self.assertGreater(biases[0][0], 0.0)

    def test_first_layer_updates_with_nonzero_weights(self):
        mlp = MLP(weights=[np.ones((1, 1)), np.ones((1, 1))],
                  biases=[np.zeros((1, 1)), np.zeros((1, 1))])
        mlp.run_input(np.ones(1))[0]
        mlp.backprop(np.ones((1,)) * 0.8)
        weights = mlp.get_weights()
        self.assertGreater(weights[0][0, 0], 0.0)
        biases = mlp.get_biases()
        self.assertGreater(biases[0][0, 0], 0.0)

    def test_XOR_backprop(self):
        hidden_weights = np.random.standard_normal((2, 2))
        output_weights = np.random.standard_normal((1, 2))
        hidden_biases = np.random.standard_normal(2)
        output_biases = np.random.standard_normal(1)
        mlp = MLP([hidden_weights, output_weights],
                  [hidden_biases, output_biases],
                  0.2)

        for i in range(20000):
            mlp.run_input(np.array([1.0, 0.0]))
            mlp.backprop(np.ones(1))

            mlp.run_input(np.array([0.0, 1.0]))
            mlp.backprop(np.ones(1))

            mlp.run_input(np.array([0.0, 0.0]))
            mlp.backprop(np.zeros(1))

            mlp.run_input(np.array([1.0, 1.0]))
            mlp.backprop(np.zeros(1))

        output = mlp.run_input(np.array([1.0, 0.0]))
        self.assertGreater(output, 0.9)

        output = mlp.run_input(np.array([0.0, 1.0]))
        self.assertGreater(output, 0.9)

        output = mlp.run_input(np.array([0.0, 0.0]))
        self.assertLess(output, 0.1)

        output = mlp.run_input(np.array([1.0, 1.0]))
        self.assertLess(output, 0.1)


if __name__ == '__main__':
    unittest.main()
