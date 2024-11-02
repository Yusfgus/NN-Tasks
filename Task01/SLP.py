import pandas as pd
import numpy as np
from myModel import myModel


class SLP(myModel):
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        super().__init__(learning_rate=learning_rate, n_epochs=n_epochs)

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
