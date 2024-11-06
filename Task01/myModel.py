import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

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
            'Positive': [TP, FP],
            'Negative': [FN, TN]
        }, index=['Positive', 'Negative'])
        
        return matrix

    def plot_decision_boundary(self, X, Y, frame, Title):
        # Clear existing widgets in the frame
        for widget in frame.winfo_children():
            widget.destroy()

        # Update the frame size to make sure it's properly sized
        frame.update_idletasks()  # Forces the frame to update its size

        # Get the dimensions of the frame after updating
        frame_width = max(frame.winfo_width(), 100)
        frame_height = max(frame.winfo_height(), 100)

        # Optionally set the frame to take half the screen width
        screen_width = frame.winfo_screenwidth()
        frame.config(width=screen_width // 2)

        # Create a Matplotlib figure and axis with constrained layout
        fig, ax = plt.subplots(constrained_layout=True)
        fig_width = max((frame_width / fig.dpi) - 1, 2)
        fig_height = max((frame_height / fig.dpi) - 1, 2)
        fig.set_size_inches(fig_width, fig_height)

        # Scatter plot of the data points with colors based on class labels
        colors = ['red' if label == 1 else 'blue' for label in Y.iloc[:, 0].values]
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=colors, s=100)

        # Add legend for each scatter color without edge color
        ax.scatter([], [], c='red', label='Class 1: Positive')
        ax.scatter([], [], c='blue', label='Class 2: Negative')

        # Calculate the decision boundary line based on weights and bias, if weights are non-zero
        x_min, x_max = X.iloc[:, 0].min(), X.iloc[:, 0].max()
        if abs(self.weights[1]) > 1e-6:  # Avoid dividing by zero or near-zero weights
            x1_values = np.linspace(x_min, x_max, 100)
            x2_values = -(self.weights[0] * x1_values + self.bias) / self.weights[1]
            ax.plot(x1_values, x2_values, color="green", label="Decision Boundary")

        # Set labels and title
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])
        ax.set_title(Title)  # Use the passed title for the plot

        # Embed the Matplotlib figure in the Tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")

        # Draw the canvas
        canvas.draw()

    def display_confusion_matrix(self, matrix, title):
        # Create a new plot for the confusion matrix
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=ax)
        ax.set_title(title)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Display the plot inline in Jupyter Notebook
        plt.show()