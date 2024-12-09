import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

class FFNN():
    def __init__(self, input_size, num_classes):
        self.num_classes = num_classes
        # Define the model
        self.model = models.Sequential([
            layers.Input(shape=(input_size,)),  # Input layer with the specified input size
            layers.Dense(128, activation='relu'),  # Hidden layer 1 (with 128 neurons)
            layers.Dense(64, activation='relu'),   # Hidden layer 2 (with 64 neurons)
            layers.Dense(num_classes, activation='softmax')  # Output layer with softmax for multi-class classification
        ])

        # Compile the model
        self.model.compile(optimizer='adam', 
                            loss='categorical_crossentropy',  # For multi-class classification
                            metrics=['accuracy'])

    def fit(self, X_train, Y_train, epochs, batch_size=32):
        # Assuming Y_train contains labels in integer form (e.g., [0, 1, 2, 3, 4])
        Y_train_categorical = to_categorical(Y_train, num_classes=self.num_classes)

        self.model.fit(X_train, Y_train_categorical, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        # Predict the labels for the test set
        predictions = self.model.predict(X)

        # If it's a multi-class classification task, get the predicted class for each sample
        predicted_classes = np.argmax(predictions, axis=1)

        return predicted_classes

    def save_h5(self, pre_method, epochs, accuracy):
        file_name = f'FFNN-m{pre_method}-e{epochs}-a{int(accuracy*100)}'
        self.model.save(f'Models/FFNN/{file_name}.h5')