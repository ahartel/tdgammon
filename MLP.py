import numpy as np
import unittest
import logging
import copy


class RandomNLayerMLP:

    def __init__(self, num_units_per_layer, learning_rate=0.1):
        num_inputs = num_units_per_layer[0]
        num_outputs = num_units_per_layer[len(num_units_per_layer) - 1]
        num_hidden_layers = len(num_units_per_layer) - 2
        self.weights = []
        self.biases = []
        for layer in range(1, len(num_units_per_layer)):
            self.weights.append(np.random.uniform(0.0, 0.1, (num_units_per_layer[layer], num_units_per_layer[layer-1])))
            self.biases.append(np.random.uniform(0.0, 0.1, num_units_per_layer[layer]))
        # self.hidden_weights = np.random.uniform(0.0, 0.1, (num_hiddens, num_inputs))
        # self.output_weights = np.random.uniform(0.0, 0.1, (num_outputs, num_hiddens))
        # self.hidden_biases = np.random.uniform(0.0, 0.1, num_hiddens)
        # self.output_biases = np.random.uniform(0.0, 0.1, num_outputs)

        self.mlp = MLP(self.weights,
                       self.biases,
                       learning_rate)

        self.num_outputs = num_outputs

    def run_input(self, inputs, save_inputs_and_activations=True):
        return self.mlp.run_input(inputs, save_inputs_and_activations)

    def backprop(self, expected):
        self.mlp.backprop(expected)

    def set_weights(self, hidden_weights, output_weights):
        self.mlp.set_weights([hidden_weights, output_weights])

    def set_biases(self, biases):
        self.mlp.set_biases(biases)

    def gradient(self):
        return self.mlp.gradient()

    def add_to_weights(self, delta):
        self.mlp.add_to_weights(delta)

    def add_to_biases(self, delta):
        self.mlp.add_to_biases(delta)


class Sigmoid:
    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x):
        s = Sigmoid.forward(x)
        return s * (1 - s)


class Identity:
    @staticmethod
    def forward(x):
        return x

    @staticmethod
    def backward(x):
        return np.ones_like(x)


class ReLU:
    @staticmethod
    def forward(x):
        x[x < 0] = 0
        return x

    @staticmethod
    def backward(x):
        x[x < 0] = 0
        x[x >= 0] = 1.0
        return x


MLP_logger = None


class MLP:
    def __init__(self, weights, biases, learning_rate=0.1, activation_function=Sigmoid):
        assert (len(weights) == len(biases))
        self.__num_layers = len(weights)
        self.__weights = weights
        self.__biases = biases
        # for layer in weights:
        #    print(layer.shape)
        self.last_activations = [None for _ in range(self.__num_layers)]
        self.last_membranes = [None for _ in range(self.__num_layers)]
        self.last_input = None
        self.learning_rate = learning_rate
        global MLP_logger
        if MLP_logger is None:
            self.logger = self.prepare_logger()
            MLP_logger = self.logger
        else:
            self.logger = MLP_logger
        self.activation = activation_function

    @staticmethod
    def prepare_logger():
        # create logger with 'spam_application'
        logger = logging.getLogger('MLP')
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler('MLP.log', mode='w')
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def set_weights(self, weights):
        assert(len(weights) == self.__num_layers)
        self.__weights = weights

    def set_biases(self, biases):
        assert(len(biases) == self.__num_layers)
        self.__biases = biases

    def run_input(self, input, save_inputs_and_activations=False):
        # self.logger.debug("Network Input: {}".format(input))
        # self.logger.debug("Input weights: {}".format(self.__weights[0]))
        last_activations = [None for _ in range(self.__num_layers)]
        last_membranes = [None for _ in range(self.__num_layers)]
        # zero is the input layer
        last_membranes[0] = self.__weights[0].dot(input) + self.__biases[0]
        last_activations[0] = self.activation.forward(last_membranes[0])
        for layer in range(1, self.__num_layers):
            # self.logger.debug("Layer {}'s input: {}".format(layer, last_activations[layer - 1]))
            # self.logger.debug("Layer {}'s weights: {}".format(layer, self.__weights[layer]))
            last_membranes[layer] = self.__weights[layer].dot(last_activations[layer - 1]) + self.__biases[layer]
            last_activations[layer] = self.activation.forward(last_membranes[layer])

        if save_inputs_and_activations:
            self.last_input = input
            self.last_activations = last_activations
            self.last_membranes = last_membranes

        # self.logger.debug("Network output: {}".format(last_activations[self.__num_layers - 1]))
        return last_activations[self.__num_layers - 1]

    def backprop(self, expected):
        weight_updates = [np.zeros_like(wgt) for wgt in self.__weights]
        bias_updates = [np.zeros_like(bias) for bias in self.__biases]
        previous_level_error = None
        for layer in range(self.__num_layers, 0, -1):
            # in this method, next_layer_output is considered from the point of view from the output layer,
            # not from the input layer.

            # recall which layer did what
            if layer > 1:
                next_layer_output = self.last_activations[layer - 2]
            else:
                next_layer_output = self.last_input

            this_layer_membranes = self.last_membranes[layer - 1]

            # calculate error signal
            if layer == self.__num_layers:
                this_layer_output = self.last_activations[layer - 1]
                # we are avoiding the storage of the "membrane potential" here
                # and also optimising the computation by calculating sigmoid' as sigmoid(1-sigmoid)
                error_signal = (expected - this_layer_output) * self.activation.backward(this_layer_membranes)
            else:
                shape = self.__weights[layer].shape
                error_signal = np.zeros(shape[1])
                derivative = self.activation.backward(this_layer_membranes)
                for k in range(shape[1]):
                    for j in range(shape[0]):
                        error_signal[k] += self.__weights[layer][j, k] * previous_level_error[j]
                    error_signal[k] *= derivative[k]
            previous_level_error = error_signal
            # print("backprop: layer= {}, error_signal= {},\nnext_layer_output= {}".format(
            #     layer,
            #     error_signal,
            #     next_layer_output
            # ))
            # calculate gradient
            delta_w = np.outer(error_signal, next_layer_output)
            weight_updates[layer - 1] = delta_w
            bias_updates[layer - 1] = error_signal

        # print("backprop: weight_updates= {}".format(weight_updates))
        # print("backprop: bias_updates= {}".format(bias_updates))

        # update
        self.add_to_weights(weight_updates)
        self.add_to_biases(bias_updates)

    def backprop_using_gradient(self, expected):
        weight_updates, bias_updates = self.gradient()
        error_signal = (expected[0] - self.last_activations[self.__num_layers-1][0])
        for layer in range(self.__num_layers, 0, -1):
            self.__weights[layer - 1] += weight_updates[layer-1] * error_signal
            self.__biases[layer - 1] += bias_updates[layer-1] * error_signal

    def gradient(self, output_neuron):
        weight_updates = [np.zeros_like(wgt) for wgt in self.__weights]
        bias_updates = [np.zeros_like(bias) for bias in self.__biases]
        if self.last_input is None:
            return weight_updates, bias_updates
        previous_level_error = None

        for layer in range(self.__num_layers, 0, -1):
            # recall which layer did what
            if layer > 1:
                prev_layer_output = self.last_activations[layer - 2]
            else:
                prev_layer_output = self.last_input

            this_layer_membranes = self.last_membranes[layer - 1]

            if layer == self.__num_layers:
                error_signal = self.activation.backward(this_layer_membranes)[output_neuron]
            elif layer == self.__num_layers-1:
                shape = self.__weights[layer].shape
                error_signal = np.zeros(shape[1])
                derivative = self.activation.backward(this_layer_membranes)
                for k in range(shape[1]):
                    error_signal[k] += self.__weights[layer][output_neuron, k] * previous_level_error
                    error_signal[k] *= derivative[k]
            else:
                shape = self.__weights[layer].shape
                error_signal = np.zeros(shape[1])
                derivative = self.activation.backward(this_layer_membranes)
                for k in range(shape[1]):
                    for j in range(shape[0]):
                        error_signal[k] += self.__weights[layer][j, k] * previous_level_error[j]
                    error_signal[k] *= derivative[k]
            previous_level_error = error_signal
            # print("gradient: layer= {}, error_signal= {},\nprev_layer_output= {}".format(
            #     layer,
            #     error_signal,
            #     prev_layer_output
            # ))
            # calculate gradient
            # is this the right order? Or do we need to transpose this? or switch args?
            # documentation says:
            # Given two vectors, a = [a0, a1, ..., aM] and b = [b0, b1, ..., bN], the
            # outer product[1] is:
            #
            # [[a0 * b0  a0 * b1...a0 * bN]
            #  [a1 * b0.
            #  [....
            #  [aM * b0            aM * bN]]
            delta_w = np.outer(error_signal, prev_layer_output)
            # update
            weight_updates[layer - 1] = delta_w
            bias_updates[layer - 1] = error_signal
        # print("gradient: weight_updates= {}".format(weight_updates))
        # print("gradient: bias_updates= {}".format(bias_updates))
        return weight_updates, bias_updates

    def get_weights(self):
        return self.__weights

    def get_biases(self):
        return self.__biases

    def add_to_weights(self, delta):
        for layer in range(self.__num_layers):
            self.__weights[layer] += self.learning_rate * delta[layer]

    def add_to_biases(self, delta):
        for layer in range(self.__num_layers):
            self.__biases[layer] += self.learning_rate * delta[layer]


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
        self.assertAlmostEqual(Sigmoid.forward(2.0), 0.8807, 3)
        self.assertAlmostEqual(Sigmoid.forward(1.0), 0.731, 3)
        self.assertAlmostEqual(Sigmoid.forward(0.0), 0.5, 3)
        self.assertAlmostEqual(Sigmoid.forward(-1.0), 0.2689, 3)
        self.assertAlmostEqual(Sigmoid.forward(-2.0), 0.1192, 3)

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
        mlp.run_input(np.ones(1), save_inputs_and_activations=True)[0]
        mlp.backprop(np.ones((1,)) * 0.8)
        weights = mlp.get_weights()
        self.assertGreater(weights[0][0, 0], 0.0)
        biases = mlp.get_biases()
        self.assertGreater(biases[0][0, 0], 0.0)

    def test_first_layer_updates_even_with_zero_weights(self):
        mlp = MLP(weights=[np.zeros((1, 1)), np.ones((1, 1))],
                  biases=[np.zeros(1), np.zeros(1)])
        mlp.run_input(np.ones(1), save_inputs_and_activations=True)
        mlp.backprop(np.ones(1) * 0.8)
        weights = mlp.get_weights()
        self.assertGreater(weights[0][0, 0], 0.0)
        biases = mlp.get_biases()
        self.assertGreater(biases[0][0], 0.0)

    def test_first_layer_updates_with_nonzero_weights(self):
        mlp = MLP(weights=[np.ones((1, 1)), np.ones((1, 1))],
                  biases=[np.zeros((1, 1)), np.zeros((1, 1))])
        mlp.run_input(np.ones(1), save_inputs_and_activations=True)[0]
        mlp.backprop(np.ones((1,)) * 0.8)
        weights = mlp.get_weights()
        self.assertGreater(weights[0][0, 0], 0.0)
        biases = mlp.get_biases()
        self.assertGreater(biases[0][0, 0], 0.0)

    def test_XOR_backprop_with_gradient(self):
        hidden_weights = np.random.standard_normal((2, 2))
        output_weights = np.random.standard_normal((1, 2))
        hidden_biases = np.random.standard_normal(2)
        output_biases = np.random.standard_normal(1)
        mlp = MLP([hidden_weights, output_weights],
                  [hidden_biases, output_biases],
                  0.2)

        for i in range(20000):
            mlp.run_input(np.array([1.0, 0.0]))
            mlp.backprop_using_gradient(np.ones(1))

            mlp.run_input(np.array([0.0, 1.0]))
            mlp.backprop_using_gradient(np.ones(1))

            mlp.run_input(np.array([0.0, 0.0]))
            mlp.backprop_using_gradient(np.zeros(1))

            mlp.run_input(np.array([1.0, 1.0]))
            mlp.backprop_using_gradient(np.zeros(1))

        output = mlp.run_input(np.array([1.0, 0.0]))
        self.assertGreater(output, 0.9)

        output = mlp.run_input(np.array([0.0, 1.0]))
        self.assertGreater(output, 0.9)

        output = mlp.run_input(np.array([0.0, 0.0]))
        self.assertLess(output, 0.1)

        output = mlp.run_input(np.array([1.0, 1.0]))
        self.assertLess(output, 0.1)

    @unittest.skip
    def test_XOR_backprop(self):
        hidden_weights = np.random.standard_normal((2, 2))
        output_weights = np.random.standard_normal((1, 2))
        hidden_biases = np.random.standard_normal(2)
        output_biases = np.random.standard_normal(1)
        mlp = MLP([hidden_weights, output_weights],
                  [hidden_biases, output_biases],
                  0.2)

        for i in range(20000):
            mlp.run_input(np.array([1.0, 0.0]), save_inputs_and_activations=True)
            mlp.backprop(np.ones(1))

            mlp.run_input(np.array([0.0, 1.0]), save_inputs_and_activations=True)
            mlp.backprop(np.ones(1))

            mlp.run_input(np.array([0.0, 0.0]), save_inputs_and_activations=True)
            mlp.backprop(np.zeros(1))

            mlp.run_input(np.array([1.0, 1.0]), save_inputs_and_activations=True)
            mlp.backprop(np.zeros(1))

        output = mlp.run_input(np.array([1.0, 0.0]))
        self.assertGreater(output, 0.9)

        output = mlp.run_input(np.array([0.0, 1.0]))
        self.assertGreater(output, 0.9)

        output = mlp.run_input(np.array([0.0, 0.0]))
        self.assertLess(output, 0.1)

        output = mlp.run_input(np.array([1.0, 1.0]))
        self.assertLess(output, 0.1)

    def test_NoGradientWithZeroWeights(self):
        mlp = MLP([np.zeros((1,))], [np.zeros((1,))])
        output = mlp.run_input(np.ones((1)), save_inputs_and_activations=True)[0]
        print(output)
        delta_w, delta_b = mlp.gradient(0)
        print(delta_w)
        self.assertAlmostEqual(delta_w[0][0][0], 0.25, 3)

    def test_GradientVsBackprop111(self):
        mlp = MLP(weights=[np.ones((1, 1)) * 0.5,
                           np.ones((1, 1)) * 0.5],
                  biases=[np.zeros((1,)),
                          np.zeros((1,))])
        initial_weights = np.copy(mlp.get_weights())
        initial_biases = np.copy(mlp.get_biases())
        print("Initial weights: {}".format(initial_weights))
        print("Initial biases: {}".format(initial_biases))
        output = mlp.run_input(np.ones(1), save_inputs_and_activations=True)[0]
        weight_updates, bias_updates = mlp.gradient(0)
        mlp.backprop([1+output])
        for layer in range(2):
            self.assertEqual(initial_weights[layer]+weight_updates[layer]*mlp.learning_rate,
                             mlp.get_weights()[layer],
                             "Weights for layer {} don't match".format(layer))
            self.assertEqual(initial_biases[layer]+bias_updates[layer]*mlp.learning_rate,
                             mlp.get_biases()[layer],
                             "Biases for layer {} don't match".format(layer))

    def test_GradientVsBackprop221(self):
        mlp = MLP(weights=[np.ones((2, 2)) * 0.5,
                           np.ones((1, 2)) * 0.5],
                  biases=[np.zeros((2,)),
                          np.zeros((1,))])
        expected_output = 0.5
        initial_weights = copy.deepcopy(mlp.get_weights())
        initial_biases = copy.deepcopy(mlp.get_biases())
        print("Initial weights: {}".format(initial_weights))
        print("Initial biases: {}".format(initial_biases))
        output = mlp.run_input(np.ones(2), save_inputs_and_activations=True)[0]
        print("Output: {}".format(output))
        weight_updates, bias_updates = mlp.gradient(0)
        print("gradient: bias_updates*error= {}".format(
            [bias_updates[0]*(expected_output-output),
             bias_updates[1]*(expected_output-output)]))
        mlp.backprop([0.5])
        print("Weights after backprop: {}".format(mlp.get_weights()))
        print("Initial weights: {}".format(initial_weights))
        for layer in range(2):
            self.assertTrue(
                np.array_equal(
                    initial_weights[layer]+weight_updates[layer]*mlp.learning_rate*(expected_output-output),
                    mlp.get_weights()[layer]),
                "Weights for layer {} don't match".format(layer))
            self.assertTrue(
                np.allclose(
                    initial_biases[layer]+bias_updates[layer]*mlp.learning_rate*(expected_output-output),
                    mlp.get_biases()[layer]),
                "Biases for layer {} don't match. updated= {}, mlp= {}".format(
                    layer,
                    initial_biases[layer] + bias_updates[layer] * mlp.learning_rate * (expected_output - output),
                    mlp.get_biases()[layer]))


if __name__ == '__main__':
    unittest.main()
