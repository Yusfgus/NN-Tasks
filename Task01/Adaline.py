import pandas as pd
import numpy as np
from myModel import myModel


class Adaline(myModel):
    def __init__(self, learning_rate=0.01, n_epochs=50, bias_bool=True, mx_mse=0):
        super().__init__(learning_rate=learning_rate, n_epochs=n_epochs)
        self.bias_bool=bias_bool
        self.mx_mse=mx_mse

    def fit(self, X, Y):
        for epoch in range(self.n_epochs):
            for i in range(X.shape[0]):
                # Calculate the weighted sum with or without bias based on bias_bool
                linear_output = np.dot(X.iloc[i].values, self.weights) + (self.bias if self.bias_bool else 0)

                y_pred = linear_output
                y = Y.iloc[i, 0]

                # Update weights and bias if there's an error
                if y_pred != y:
                    error = y - y_pred
                    self.weights += self.learning_rate * error * X.iloc[i].values
                    if self.bias_bool:  # Update bias only if bias_bool is True
                        self.bias += self.learning_rate * error
            
            total_error = 0
            for i in range(X.shape[0]):
                # Calculate the weighted sum with or without bias based on bias_bool
                linear_output = np.dot(X.iloc[i].values, self.weights) + (self.bias if self.bias_bool else 0)

                y_pred = linear_output
                y = Y.iloc[i, 0]

                # Calculate MSE
                if y_pred != y:
                    error = y - y_pred
                    total_error += error**2

            MSE = (total_error / 2) / X.shape[0]
            if MSE <= self.mx_mse:
                print(f"break in epoch #{epoch}")
                return
