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
        self.layers = layers # 5 3 4 3
        self.learning_rate = learning_rate
        self.activation_name = activation
        self.epochs = epochs
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]) * 0.1)
            self.biases.append(np.random.randn(1, layers[i + 1]) * 0.1)

        # print(f"weights => \n {self.weights} \n Biases => \n {self.biases}")

        # Set activation function
        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = self.tanh
            self.activation_derivative = self.tanh_derivative
        else:
            raise ValueError("Unsupported activation function. Use 'sigmoid' or 'tanh'.")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.power(x, 2)

    def fit(self, X, y):
        print(f"weights => \n {self.weights} \n Biases => \n {self.biases}")

        """
        Train the MLP using backpropagation.
        :param X: Training data (features).
        :param y: Training labels (target).
        :return: Best weights (after training).
        """
        X = np.array(X)
        y = np.array(y)
        y_one_hot = np.eye(self.layers[-1])[y]  # Convert y to one-hot encoding
        print("self.layers : ", self.layers)
        print("y_one_hot : " , y_one_hot)
        for epoch in range(self.epochs):
            # Forward pass
            activations = [X]
            for w, b in zip(self.weights, self.biases):
                activations.append(self.activation(np.dot(activations[-1], w) + b))

            # Backward pass
            deltas = [activations[-1] - y_one_hot]  # Error at output layer
            for i in range(len(self.layers) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[i].T) * self.activation_derivative(activations[i]))
            deltas.reverse()

            # Gradient descent update
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * activations[i].T.dot(deltas[i])
                self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

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



# Example usage
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Preprocessing import preprocessing
from Preprocessing import pre_bird_category


all_data = pd.read_csv(r"D:\slides\4th grade\1st term\NN\GIT - NN-Tasks\NN-Tasks\Task02\birds.csv")
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
model.calculate_accuracy_and_confusion_matrix(Xtrain, ytrain, Xtest, ytest)


# model = MLP([3, 4], 0.01, activation='sigmoid', epochs=1000)
# best_weights = model.fit(Xtrain, ytrain)


# model.calculate_accuracy_and_confusion_matrix(Xtrain, ytrain, Xtest, ytest,)
