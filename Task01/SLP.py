import pandas as pd
import numpy as np
from myModel import myModel


class SLP(myModel):
    def __init__(self, learning_rate=0.01, n_epochs=1000, bias_bool=True):
        # Only pass arguments that myModel expects
        super().__init__(learning_rate=learning_rate, n_epochs=n_epochs)
        self.bias_bool = bias_bool  # Set bias_bool here, without passing it to myModel
        self.bias = 0.0  # Initialize bias if necessary

    def fit(self, X, Y):
        for epoch in range(self.n_epochs):
            for i in range(X.shape[0]):
                # Calculate the weighted sum with or without bias based on bias_bool
                linear_output = np.dot(X.iloc[i].values, self.weights) + (self.bias if self.bias_bool else 0)
                print(self.bias_bool)
                # Apply signum function
                y_pred = 1 if linear_output >= 0 else -1
                y = Y.iloc[i, 0]

                # Update weights and bias if there's an error
                if y_pred != y:
                    error = y - y_pred
                    self.weights += self.learning_rate * error * X.iloc[i].values
                    if self.bias_bool:  # Update bias only if bias_bool is True
                        self.bias += self.learning_rate * error
