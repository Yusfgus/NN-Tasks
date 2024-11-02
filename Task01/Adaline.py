import pandas as pd
import numpy as np
from myModel import myModel


class Adaline(myModel):
    def __init__(self, learning_rate=0.01, n_epochs=50):
        super().__init__(learning_rate=learning_rate, n_epochs=n_epochs)

    def fit(self, X, Y):
        # implement me please :(
        pass
