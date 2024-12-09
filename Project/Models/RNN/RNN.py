import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout

class RNN():
    def __init__(self, max_words, embedding_dim, max_sequence_length, num_classes):
        self.num_classes = num_classes
        # Define the model
        self.model = models.Sequential([
            layers.Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length),  # Embedding layer
            layers.SimpleRNN(128, activation='tanh', return_sequences=False),  # RNN layer (can replace with LSTM or GRU)
            layers.Dense(64, activation='relu'),  # Fully connected layer
            layers.Dense(num_classes, activation='softmax')  # Output layer
        ])

        # Compile the model
        self.model.compile(optimizer='adam',
                            loss='categorical_crossentropy',  # For multi-class classification
                            metrics=['accuracy'])

        # Summary of the model
        # self.model.summary()

    def fit(self, X_train_padded, Y_train, epochs=10, batch_size=32, validation_split=0.2):
        # Assuming Y_train contains labels in integer form (e.g., [0, 1, 2, 3, 4])
        Y_train_categorical = to_categorical(Y_train, num_classes=self.num_classes)

        self.model.fit(X_train_padded, Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def predict(self, X):
        # Predict the labels for the test set
        predictions = self.model.predict(X)

        # If it's a multi-class classification task, get the predicted class for each sample
        predicted_classes = np.argmax(predictions, axis=1)

        return predicted_classes

    def save_h5(self, pre_method, epochs, accuracy):
        file_name = f'RNN-m{pre_method}-e{epochs}-a{int(accuracy*100)}'
        self.model.save(f'Models/RNN/{file_name}.h5')