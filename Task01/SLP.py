import pandas as pd
import numpy as np


class SLP():
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        # Perceptron parameters
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        # Initialize weights and bias
        np.random.seed(50)
        self.weights = np.random.rand(2)
        self.bias = np.random.rand()

    def fit(self, X, Y):
        for epoch in range(self.n_epochs):
            for i in range(X.shape[0]):
                # Calculate the weighted sum
                linear_output = np.dot(X.iloc[i].values, self.weights) + self.bias

                # Apply signum function
                y_pred = 1 if linear_output >= 0 else -1
                y = Y.iloc[i, 0]

                # Update weights and bias if there's an error
                if y_pred != y:
                    error = y - y_pred
                    self.weights += self.learning_rate * error * X.iloc[i].values
                    self.bias += self.learning_rate * error

    def predict(self, X, column_name):

        Y_pred = []
        for i in range(X.shape[0]):
            # Calculate the weighted sum
            linear_output = np.dot(X.iloc[i].values, self.weights) + self.bias

            # Apply signum function
            y_pred = 1 if linear_output >= 0 else -1

            Y_pred.append([y_pred])

        return pd.DataFrame(Y_pred, columns=[column_name])

    def accuracy_score(self, Y, Y_predict):
        # Calculate the number of correct predictions
        correct_predictions = (Y.values.flatten() == Y_predict.values.flatten()).sum()
        
        # Calculate accuracy
        accuracy = correct_predictions / len(Y)

        return accuracy

    def confusion_matrix(self, Y, Y_pred):
        TP, FP = 0, 0
        FN, TN = 0, 0
        for i in range(Y.shape[0]):
            y = Y.iloc[i, 0]
            y_pred = Y_pred.iloc[i, 0]

            if y_pred == 1:
                if y == 1: TP+=1 
                else: FP+=1
            elif y_pred == -1:
                if y == 1: FN+=1 
                else: TN+=1

        matrix = df = pd.DataFrame({
            'Predicted Positive': [TP, FP],
            'Predicted Negative': [FN, TN]
        }, index=['Actual Positive', 'Actual Negative'])
        
        return matrix

