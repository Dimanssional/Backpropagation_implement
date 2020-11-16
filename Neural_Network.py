import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# import pandas as pd

np.random.seed(100)


class Dense:

    def __init__(self, input_dims, neuron_count, activation=None):
        self.neuron_count = neuron_count
        self.weights = np.random.normal(0.0, np.sqrt(2/(input_dims * neuron_count)), (input_dims, neuron_count))

        self.layer_activation = None
        self.activation = activation
        self.biases = np.zeros(neuron_count)

        self.mse = None
        self.change_W = None

    def activation_fun(self, x):
        if self.activation == 'ReLU':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)

    def derivate_activation(self, x):
        if self.activation == 'sigmoid':
            fx = 1 / (1 + np.exp(-x))
            return fx * (1 - fx)
        elif self.activation == 'tanh':
            return -((2*(np.exp(2*x) - 1) * np.exp(2*x)) / (np.exp(2*x) + 1)**2) + (2 * np.exp(2*x) / (np.exp(2*x) + 1))
        elif self.activation == 'ReLU':
            x[x <= 0] = 0
            x[x > 0] = 1
            return x

    def activate_neurons(self, x):
        y_pred = np.dot(x, self.weights) + self.biases
        self.layer_activation = self.activation_fun(y_pred)
        return self.layer_activation


class NeuralNetwork:

    def __init__(self):
        self.nn_layers = []

    def add_layer(self, neuron_count, inp_shape=None, activation=None):
        if len(self.nn_layers) == 0 and inp_shape is None:
            raise ValueError("Please enter input shape for first layer!")

        if inp_shape is None:
            inp_shape = self.nn_layers[-1].neuron_count

        self.nn_layers.append(Dense(inp_shape, neuron_count, activation))

    def feed_forward(self, X):

        for l in self.nn_layers:
            X = l.activate_neurons(X)
        return X

    def predict(self, X):
        result = self.feed_forward(X)
        if result.ndim == 1:
            return np.argmax(result)
        return np.argmax(result, axis=1)

    def backward(self, X, y_true, learning_rate):
        y_pred = self.feed_forward(X)

        for l in reversed(range(len(self.nn_layers))):
            nn_layer = self.nn_layers[l]

            if nn_layer is self.nn_layers[-1]:
                nn_layer.mse = y_true - y_pred
                nn_layer.change_W = (nn_layer.mse * nn_layer.derivate_activation(y_pred))
            else:
                output_layer = self.nn_layers[l + 1]
                nn_layer.mse = np.dot(output_layer.weights, output_layer.change_W)
                nn_layer.change_W = (nn_layer.mse * nn_layer.derivate_activation(nn_layer.layer_activation))

        for i in range(len(self.nn_layers)):
            nn_layer = self.nn_layers[i]

            if i == 0:
                inputs = np.array(X, ndmin=2)
            else:
                inputs = np.array(self.nn_layers[i - 1].layer_activation, ndmin=2)

            nn_layer.weights += learning_rate * nn_layer.change_W * np.transpose(inputs)

    def mse_loss(self, y_true, y_pred):
        return ((y_true - y_pred)**2).mean()

    def fit(self, x, y_true, learning_rate, epochs):
        errors = []

        for e in range(epochs):
            for i in range(len(x)):
                self.backward(x[i], y_true[i], learning_rate)
            if e % 10 == 0:
                loss = self.mse_loss(y_true, my_NN.predict(x))
                errors.append(loss)
                print(f"epoch: {e}, loss: {loss:2f}")
        return errors

    def plotting_train(self, fit):
        plt.plot(fit)
        plt.title('Neural Network Training')
        plt.xlabel('Train Epoch')
        plt.ylabel('Train Loss')
        plt.show()


if __name__ == "__main__":

    dataset = load_iris()
    # mnist_dataset_train = pd.read_csv('dataset/mnist_train_100.csv')
    # mnist_dataset_test = pd.read_csv('dataset/mnist_test_10.csv')
    # X_mnist_train = mnist_dataset_train
    # X_mnist_test = mnist_dataset_test
    #
    # all_values = X_mnist_train[1].split(',')
    # image_array = np.asfarray(all_values[1:]).reshape((28, 28))
    #
    # all_values_test = X_mnist_test[1].split(',')
    # image_array_test = np.asfarray(all_values[1:]).reshape((28, 28))
    # print(dataset.data)
    # print(dataset.target)
    # print(dataset.target_names)

    X = dataset.data[0:65]
    Y = dataset.target[0:65]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

    my_NN = NeuralNetwork()
    # please enter input shape like a number of attributes in dataset(for iris 4, for mnist 784)
    my_NN.add_layer(65, 4, activation='ReLU')
    my_NN.add_layer(130, activation='tanh')
    my_NN.add_layer(260, activation='tanh')
    # my_NN.add_layer(200, activation='tanh')
    # my_NN.add_layer(200, activation='tanh')
    my_NN.add_layer(500, activation='tanh')
    my_NN.add_layer(400, activation='tanh')
    # my_NN.add_layer(150, activation='ReLU')
    # my_NN.add_layer(100, activation='ReLU')
    # my_NN.add_layer(60, activation='tanh')
    my_NN.add_layer(3, activation='sigmoid')
    # this topology giving the best accuracy for iris_dataset, experimentaly

    training = my_NN.fit(X_train, y_train, 0.001, 101)
    print(f'accuracy: {(accuracy_score(my_NN.predict(X_test), y_test.flatten()) * 100)}%')

    my_NN.plotting_train(training)

































# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
#
# np.random.seed(100)
#
# class Dense:
#
#     def __init__(self, inp_dims, neuron_count, activation=None):
#         self.neuron_count = neuron_count
#         self.weights = np.random.normal(loc=0.0, scale=np.sqrt(2/(inp_dims * neuron_count)), size=(inp_dims, neuron_count))
#         self.biases = np.ones(neuron_count, dtype=float)
#
#         self.layer_activation = None
#         self.activation = activation
#
#         self.error = None
#         self.change_constant = None
#
#     def get_neuron_count(self):
#         return self.neuron_count
#
#     def activation_fun(self, x):
#         if self.activation == 'sigmoid':
#             return 1 / (1 + np.exp(-x))
#         elif self.activation == 'ReLU':
#             return np.maximum(0, x)
#         elif self.activation == 'tanh':
#             return (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
#         elif self.activation == 'softmax':
#             return np.exp(x) / sum(np.exp(x))
#
#     def derivate_activation(self, x):
#         if self.activation == 'sigmoid':
#             fx = 1 / (1 + np.exp(-x))
#             return fx * (1 - fx)
#         elif self.activation == 'ReLU':
#             x[x <= 0] = 0
#             x[x > 0] = 1
#             return x
#         elif self.activation == 'tanh':
#             return -((2*(np.exp(2*x) - 1) * np.exp(2*x)) / (np.exp(2*x) + 1)**2) + (2 * np.exp(2*x) / (np.exp(2*x) + 1))
#             # return 1 / np.cosh(x)**2
#         elif self.activation == 'softmax':
#             # return np.exp(x) / sum(np.exp(x))
#             pass
#         # in the next time
#
#     def activate_neurons(self, x):
#         output = np.dot(x, self.weights) + self.biases
#         self.layer_activation = self.activation_fun(output)
#         return self.layer_activation
#
#
# class NeuralNetwork:
#
#     def __init__(self):
#         self.layers_nn = []
#
#     def add_layer(self, neuron_count, inp_shape=None, activation=None):
#         if len(self.layers_nn) == 0 and inp_shape == None:
#             raise ValueError("Must define input shape for first layer")
#
#         if inp_shape == None:
#             inp_shape = self.layers_nn[-1].neuron_count
#         self.layers_nn.append(Dense(inp_shape, neuron_count, activation))
#
#     def feed_forward(self, x):
#         for i in self.layers_nn:
#             x = i.activate_neurons(x)
#         return x
#
#     def predict(self, x):
#         result = self.feed_forward(x)
#         if result.ndim is 1:
#             return np.argmax(result)
#         return np.argmax(result, axis=1)
#
#     def mse_loss(self, y_pred, y_true):
#         return ((y_true - y_pred)**2).mean()
#
#     def backward(self, x, y_true, learning_rate):
#         y_pred = self.feed_forward(x)
#
#         for l in reversed(range(len(self.layers_nn))):
#
#             nn_layer = self.layers_nn[l]
#
#             if nn_layer == self.layers_nn[-1]:
#                 nn_layer.error = y_true - y_pred
#                 nn_layer.change_constant = (nn_layer.derivate_activation(y_pred) * nn_layer.error)
#             else:
#                 next_layer = self.layers_nn[l + 1]
#                 nn_layer.error = np.dot(next_layer.weights, next_layer.change_constant)
#                 nn_layer.change_constant = nn_layer.derivate_activation(nn_layer.layer_activation) * nn_layer.error
#
#         for i in range(len(self.layers_nn)):
#             nn_layer = self.layers_nn[i]
#
#             inputs = np.array(x if i == 0 else self.layers_nn[i - 1].layer_activation, ndmin=2)
#             nn_layer.weights += np.transpose(inputs) * nn_layer.change_constant * learning_rate
#
#     def fit(self, x, y_true, learning_rate, epochs):
#         errors = []
#         for e in range(epochs):
#             for i in range(len(x)):
#                 self.backward(x[i], y_true[i], learning_rate)
#                 loss = np.mean(np.square(y_true - My_NN.predict(x)))
#                 errors.append(loss)
#             if e % 10 == 0:
#                 print(f"epochs:{e}, loss:{loss}")
#         return errors
#
#
# if __name__ == '__main__':
#     My_NN = NeuralNetwork()
#     data = np.array([
#         [-2, -1],  # Alice
#         [25, 6],  # Bob
#         [17, 4],  # Charlie
#         [-15, -6],  # Diana
#         [30, 8],
#         [-4, -3],
#         [-10, -2],
#         [18, 6],
#         [25, 3],
#         [-1, 3],
#         [-4, -2],
#         [16, 4],
#         [5, 2],
#         [-10, -7],
#         [35, 15],
#     ])
#
#     all_y_trues = np.array([
#         [1],  # Alice
#         [0],  # Bob
#         [0],  # Charlie
#         [1],  # Diana
#         [0],
#         [1],
#         [1],
#         [0],
#         [0],
#         [1],
#         [1],
#         [0],
#         [1],
#         [1],
#         [0],
#     ])
#     dataset = load_iris()
#
#     # print(dataset.data)
#     # print(dataset.target)
#     # print(dataset.target_names)
#
#     X = dataset.data
#     Y = dataset.target
#
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#
#     x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#     y = np.array([[0], [0], [0], [1]])
#
#     My_NN.add_layer(4, 4, activation='sigmoid')
#     My_NN.add_layer(4, activation='tanh')
#     My_NN.add_layer(4, activation='tanh')
#     My_NN.add_layer(4, activation='tanh')
#     My_NN.add_layer(4, activation='tanh')
#     My_NN.add_layer(3, activation='sigmoid')
#
#     train = My_NN.fit(X_train, y_train, 0.02, 100)
#
#     print(f"Accuracy: {accuracy_score(My_NN.predict(X_test), y_test.flatten()) * 100}")
#     plt.plot(train)
#     plt.title('Changes in MSE')
#     plt.xlabel('Epoch')
#     plt.ylabel('MSE')
#     plt.show()
