import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

class MLP:
    def __init__(self, layers, learning_rate=0.01, activation='sigmoid', epochs=1000):
        """
        Initialize the Multi-Layer Perceptron.
        :param layers: List containing the number of neurons in each layer, e.g., [input_size, hidden_size, output_size].
        :param learning_rate: Learning rate for weight updates.
        :param activation: Activation function to use ('sigmoid', 'tanh').
        :param epochs: Number of epochs for training.
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation_name = activation
        self.epochs = epochs
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        self._initialize_weights()

        # Set activation function
        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = self.tanh
            self.activation_derivative = self.tanh_derivative
        else:
            raise ValueError("Unsupported activation function. Use 'sigmoid' or 'tanh'.")

    def _initialize_weights(self):
        """Initialize weights and biases for each layer."""
        for i in range(len(self.layers) - 1):
            self.weights.append(2 * np.random.rand(self.layers[i], self.layers[i + 1]) - 1)
            self.biases.append(2*np.random.rand(1, self.layers[i + 1]) -1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.power(x, 2)

    def forward_propagation(self, X):
        """Perform forward propagation through the network."""
        activations = [X]
        for w, b in zip(self.weights, self.biases):
            activations.append(self.activation(np.dot(activations[-1], w) + b))
        return activations

    def backward_propagation(self, X, y_one_hot, activations):
        """Perform backward propagation to compute the deltas."""
        deltas = []

        # Calculate delta for the output layer
        output_layer_net = activations[-1]
        delta_output = ( output_layer_net-y_one_hot ) 
        deltas.append(delta_output)

        # Calculate deltas for hidden layers
        for i in range(len(self.layers) - 2, 0, -1):  # Loop through hidden layers in reverse
            delta_next = deltas[-1]
            w_next = self.weights[i]  # Weights of the next layer
            net_hidden = activations[i]  # f(net1)
            delta_hidden = delta_next.dot(w_next.T) * self.activation(net_hidden) * (1 - self.activation(net_hidden))
            deltas.append(delta_hidden)

        # Reverse the deltas to match layer order
        deltas.reverse()
        return deltas

    def update_weights(self, activations, deltas):
        """Update weights and biases using the computed deltas."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * activations[i].T.dot(deltas[i])
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    def fit(self, X, y):
        """
        Train the MLP using backpropagation.
        :param X: Training data (features).
        :param y: Training labels (target).
        :return: Best weights (after training).
        """
        X = np.array(X)
        y = np.array(y)
        y_one_hot = np.eye(self.layers[-1])[y]  # Convert y to one-hot encoding

        for epoch in range(self.epochs):
            # Forward pass
            activations = self.forward_propagation(X)

            # Backward pass
            deltas = self.backward_propagation(X, y_one_hot, activations)

            # Update weights and biases
            self.update_weights(activations, deltas)

            # Logging training progress
            if (epoch + 1) % 100 == 0:
                predictions = np.argmax(activations[-1], axis=1)
                accuracy = accuracy_score(y, predictions)
                print(f"Epoch {epoch + 1}/{self.epochs} - Accuracy: {accuracy:.4f}")

        return self.weights

    def predict(self, X):
        """
        Predict output for given input.
        :param X: Input data.
        :return: Predicted class labels.
        """
        X = np.array(X)
        for w, b in zip(self.weights, self.biases):
            X = self.activation(np.dot(X, w) + b)
        return np.argmax(X, axis=1)

    def calculate_accuracy_and_confusion_matrix(self, X_train, y_train, X_test, y_test):
        """
        Calculate accuracy and confusion matrix for training and test sets.
        :param X_train: Training data (features).
        :param y_train: Training labels.
        :param X_test: Test data (features).
        :param y_test: Test labels.
        """
        y_train_pred = self.predict(X_train)
        y_test_pred = self.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print("Training Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        print("\nConfusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred))

    def save(self, filename):
        """
        Save weights and biases to a file.
        :param filename: Name of the file to save the weights and biases.
        """
        with open(filename, 'w') as file:
            # Save weights and biases
            for w, b in zip(self.weights, self.biases):
                np.savetxt(file, w, delimiter=",")
                file.write("\n")  # Separate weight matrices with a newline
                np.savetxt(file, b, delimiter=",")
                file.write("\n\n")  # Separate weight/bias pairs with two newlines

    def load(self, filename):
        """
        Load weights and biases from a file.
        :param filename: Name of the file to load the weights and biases from.
        """
        weights_loaded = []
        biases_loaded = []
        
        with open(filename, 'r') as file:
            content = file.read().strip().split("\n\n")
            for i in range(0, len(content), 2):
                # Load weight and bias pair
                weight_matrix = np.loadtxt(content[i].splitlines(), delimiter=',')
                bias_matrix = np.loadtxt(content[i + 1].splitlines(), delimiter=',')
                weights_loaded.append(weight_matrix)
                biases_loaded.append(bias_matrix)

        # Assign loaded weights and biases back to the model
        self.weights = weights_loaded
        self.biases = biases_loaded
        print("Weights and biases loaded successfully.")


        

# Example usage
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Preprocessing import preprocessing
from Preprocessing import pre_bird_category


all_data = pd.read_csv(r"D:\4th year\7th term\ANN\Project\NN-Tasks\Task02\birds.csv")
# preprocessing
preprocessing(data=all_data, classes=['A', 'B', 'C']) # A->0 , B->1 , C->2

# Class A Portion
Class_A = all_data.iloc[0:50, :]
# Class B Portion
Class_B = all_data.iloc[50:100, :]
# Class C Portion
Class_C = all_data.iloc[100:150, :]

# Shuffling portions
Class_A = Class_A.sample(frac=1, random_state=42).reset_index(drop=True)
Class_B = Class_B.sample(frac=1, random_state=42).reset_index(drop=True)
Class_C = Class_C.sample(frac=1, random_state=42).reset_index(drop=True)

# Data Slicing (Training & Test) for each class

TrainClass_A = Class_A.iloc[0:30, :]
TestClass_A = Class_A.iloc[30:50, :]

TrainClass_B = Class_B.iloc[0:30, :]
TestClass_B = Class_B.iloc[30:50, :]

TrainClass_C = Class_C.iloc[0:30, :]
TestClass_C = Class_C.iloc[30:50, :]


# Compining Train samples of each class
train = pd.concat([TrainClass_A, TrainClass_B, TrainClass_C])
# Compining Test samples of each class
test = pd.concat([TestClass_A, TestClass_B, TestClass_C])

# Shuffling Train and Test data
train = train.sample(frac=1, random_state=42).reset_index(drop=True)
test = test.sample(frac=1, random_state=42).reset_index(drop=True)

# split to X and y
Xtrain = train.iloc[:, :-1]
ytrain = train.iloc[:, -1]

Xtest = test.iloc[:, :-1]
ytest = test.iloc[:, -1]


#Xtrain = pd.DataFrame(np.random.rand(100, 5))  # 100 samples, 5 features
#ytrain = np.random.randint(0, 3, size=100)  # 100 samples, 3 classes

model = MLP(layers=[5, 3, 4, 3], learning_rate=0.01, activation='sigmoid', epochs=1000)
best_weights = model.fit(Xtrain, ytrain)

#model.load("mlp_weights.txt")

model.calculate_accuracy_and_confusion_matrix(Xtrain, ytrain, Xtest, ytest)

# model.save("mlp_weights4.txt")


# model = MLP([5,5,3], 0.001, activation='tanh', epochs=5000)
# best_weights = model.fit(Xtrain, ytrain)


# model.calculate_accuracy_and_confusion_matrix(Xtrain, ytrain, Xtest, ytest,)
# model.save("mlp_tanh4.txt")