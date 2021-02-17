import numpy as np
import yaml
# from functools import wraps


# class MyWrappers:
#     @staticmethod
#     def clean_vectors(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             new_args = []
#             for arg in args:
#                 if isinstance(arg, np.ndarray):
#                     if len(arg.shape) == 1:
#                         arg = arg[None, :]  # make row vector
#                 new_args.append(arg)
#             return func(*new_args, **kwargs)
#         return wrapper


class NeuralNetwork(object):
    def __init__(self,
                 input_nodes,
                 hidden_nodes,
                 output_nodes,
                 learning_rate,
                 hidden_activation='sigmoid',
                 hidden_activation_prime=None,
                 output_activation='self',
                 output_activation_prime=None):
        """ Custom Neural Network initialization for a single layer network

            Parameters
            ----------
            input_nodes : int
                number of input nodes
            hidden_nodes : int
                number of hidden nodes
            output_nodes : int
                number of output nodes
            learning_rate : float
                rate to train the network, recommended 0<x<=1
            hidden_activation : str or callable
                activation function to use for all nodes within the hidden
                network. If string, must be within {'sigmoid'}, Default:
                'sigmoid'. Note: if callable is specified,
                activation_function_prime *must* be specified.
                Available Options:
                    'sigmoid' : uses NeuralNetwork.sigmoid
                    'self' : uses NeuralNetwork.self_activation
            hidden_activation_prime : callable
                Only used when a callable is specified for the
                activation_function parameter assigns the derivative to the
                activation_function to the  activation_function_prime method
            output_activation : str or callable
                Same as hidden_activation for the output layer
            output_activation_prime : callable
                Same as hidden_activation_prime for the output layer
        """
        def evaluate_activation(act_func, prime_func):
            if act_func == 'sigmoid':
                a_func = self.sigmoid
                a_func_prime = self.sigmoid_prime
            elif act_func == 'self':
                a_func = self.self_activation
                a_func_prime = self.self_activation_prime
            else:
                a_func = act_func
                a_func_prime = prime_func
            return a_func, a_func_prime

        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(
            0.0,
            self.input_nodes ** -0.5,
            (self.input_nodes, self.hidden_nodes)
        )

        self.weights_hidden_to_output = np.random.normal(
            0.0,
            self.hidden_nodes ** -0.5,
            (self.hidden_nodes, self.output_nodes)
        )
        self.lr = learning_rate

        # ### DONE: Set self.activation_function to your implemented sigmoid
        #           function ####
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        # Replace 0 with your sigmoid calculation.
        #         self.activation_function = lambda x : 0
        self.activation_function, self.activation_function_prime = (
            evaluate_activation(hidden_activation, hidden_activation_prime)
        )
        self.output_act_func, self.output_act_func_prime = (
            evaluate_activation(output_activation, output_activation_prime)
        )

        # ## If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your
        # implementation there instead.
        #
        # def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        # self.activation_function = sigmoid

    # @MyWrappers.clean_vectors
    def train(self, features, targets):
        """ Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a
                feature
            targets: 1D array of target values
        """
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            # Implement the forward pass function below
            final_outputs, hidden_outputs = self.forward_pass_train(X)

            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(
                final_outputs, hidden_outputs,
                X, y,
                delta_weights_i_h, delta_weights_h_o
            )
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    # @MyWrappers.clean_vectors
    def forward_pass_train(self, X):
        """ Implement forward pass here

            Arguments
            ---------
            X: features batch

        """
        # ### Implement the forward pass here ####
        # ## Forward pass ###
        # DONE: Hidden layer - Replace these values with your calculations.
        # signals into hidden layer
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)

        # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # DONE: Output layer - Replace these values with your calculations.

        # signals into final output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)

        # signals from final output layer
        final_outputs = self.output_act_func(final_inputs)

        return final_outputs, hidden_outputs

    # @MyWrappers.clean_vectors
    def backpropagation(self,
                        final_outputs, hidden_outputs,
                        X, y,
                        delta_weights_i_h, delta_weights_h_o):
        """ Implement backpropagation

            Arguments
            ---------
            final_outputs: output from forward pass
            X : np.ndarray (nxm)
                input values where
                n is the number of records and
                m is the number of features
            y: target (i.e. label) batch
            hidden_outputs : np.ndarray
                hidden output values
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

            Return
            ------
            delta_weights_i_h : np.array
                updated change in weights from input to hidden layers
            delta_weights_h_o : np.array
                 updated change in weights from hidden to output layers
        """
        # ### Implement the backward pass here ####
        # ## Backward pass ###

        # DONE: Output error - Replace this value with your calculations.
        # Output layer error is the difference between desired target and actual
        #   output.
        error = y - final_outputs

        # DONE: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(error, self.weights_hidden_to_output.T)

        # DONE: Backpropagated error terms - Replace these values with your
        #   calculations.
        output_error_term = (
            error * self.output_act_func_prime(final_outputs)
        )

        hidden_error_term = (
            hidden_error * self.activation_function_prime(hidden_outputs)
        )

        # Weight step (hidden to output)
        delta_weights_h_o += np.dot(hidden_outputs[:, None],
                                    output_error_term[None, :])
        # Weight step (input to hidden)
        delta_weights_i_h += np.dot(np.array(X)[:, None],
                                    hidden_error_term[None, :])

        return delta_weights_i_h, delta_weights_h_o

    # @MyWrappers.clean_vectors
    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        """ Update weights on gradient descent step

            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        """
        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += (
                self.lr * delta_weights_h_o / n_records
        )
        # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += (
                self.lr * delta_weights_i_h / n_records
        )

    # @MyWrappers.clean_vectors
    def run(self, features):
        """ Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values

            Return
            ------
            final_outputs : np.array (nx1)
                Output from constructed neural network given instance weights
                (weights_input_to_hidden, weights_hidden_to_output) and
                activation_function
        """

        # ### Implement the forward pass here ####
        # DONE: Hidden layer - replace these values with the appropriate calculations.

        # signals into hidden layer
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)

        # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # DONE: Output layer - Replace these values with your calculations.

        # signals into final output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)

        # signals from final output layer
        final_outputs = self.output_act_func(final_inputs)

        return final_outputs

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def sigmoid_prime(cls, y=None, x=None):
        if y is None:
            y = cls.sigmoid(x)
        return y * (1 - y)

    @staticmethod
    def self_activation(x):
        return x

    @staticmethod
    def self_activation_prime(x):
        return 1

    def save_network(self, file):
        """
        Save trained network into a yaml file

        Parameters
        ----------
        file : str, or str-like
            file location
        """
        def gen_dict():
            return {
                'weights_input_to_hidden': self.weights_input_to_hidden.tolist(),
                'weights_hidden_to_output': self.weights_hidden_to_output.tolist(),
                'input_nodes': self.input_nodes,
                'hidden_nodes': self.hidden_nodes,
                'output_nodes': self.output_nodes,
                'learning_rate': self.lr,
            }

        with open(file, 'w') as fh:
            yaml.safe_dump(gen_dict(), fh)

    @classmethod
    def load_network(cls, file):
        """Generate a new NeuralNetwork instance provided file contents
        Must be in format provided by instance
        Parameters
        ----------
        file : str, or str-like
            file location

        Returns
        -------
        NeuralNetwork
            With provided architecture and weights
        """
        with open(file, 'r') as fh:
            d = yaml.safe_load(fh)
        loaded = cls(d['input_nodes'],
                     d['hidden_nodes'],
                     d['output_nodes'],
                     d['learning_rate'])
        loaded.weights_input_to_hidden = np.array(d['weights_input_to_hidden'])
        loaded.weights_hidden_to_output = np.array(
            d['weights_hidden_to_output']
        )
        return loaded


# ########################################################
# Set your hyperparameters here
# #########################################################
iterations = 1360
learning_rate = 1.6
hidden_nodes = 7
output_nodes = 1
# 0.305 = 1000,1,6,1
# 0.250 = 1500,1,6,1 - looks like dec is performing poorly
# 0.211 = 3000,1,6,1
# 0.156 = 10000,1,6,1
# 0.149 = 50000,1,6,1
# 0.134 = 8000,2,6,1 - now 0.231
# TEST = 1360,1,7,1.6