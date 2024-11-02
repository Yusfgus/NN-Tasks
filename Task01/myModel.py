import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class myModel():
    def __init__(self, learning_rate, n_epochs):
        # Perceptron parameters
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        # Initialize weights and bias
        np.random.seed(50)
        self.weights = np.random.rand(2)
        self.bias = np.random.rand()

    def fit(self, X, Y):
        pass

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

    def plot_decision_boundary(self, X, Y):
        plt.figure(figsize=(7, 5))
        # Scatter plot of the data points with colors based on class labels
        colors = ['red' if label == 1 else 'blue' for label in Y.iloc[:, 0].values]
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=colors, marker='o', edgecolor='k', s=100)
        
        # Calculate the line based on weights and bias
        
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])
        plt.title("Decision Boundary")
        plt.legend()
        plt.show()

