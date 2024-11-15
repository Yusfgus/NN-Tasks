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

    def plot_decision_boundary(self, X_train, Y_train, X_test, Y_test,Plot_frame, Title):
        # Clear existing widgets in the frame
        for widget in Plot_frame.winfo_children():
            widget.destroy()

        Plot_frame.update_idletasks()
        
        # Get dimensions of the frame
        frame_width = max(Plot_frame.winfo_width(), 100)
        frame_height = max(Plot_frame.winfo_height(), 100)

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)
        fig_width = max((frame_width / fig.dpi) - 2, 2)
        fig_height = max((frame_height / fig.dpi) - 1, 2)
        fig.set_size_inches(fig_width, fig_height)



        # Plot train dataset
        colors_train = ['red' if label == 1 else 'blue' for label in Y_train.iloc[:, 0].values]
        ax1.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=colors_train, s=100)
        ax1.scatter([], [], c='red', label='Class 1: Positive')
        ax1.scatter([], [], c='blue', label='Class 2: Negative')
        
        x_min_train, x_max_train = X_train.iloc[:, 0].min(), X_train.iloc[:, 0].max()
        if abs(self.weights[1]) > 1e-6:
            x1_values = np.linspace(x_min_train, x_max_train, 100)
            x2_values = -(self.weights[0] * x1_values + self.bias) / self.weights[1]
            ax1.plot(x1_values, x2_values, color="green", label="Decision Boundary")

        ax1.set_xlabel(X_train.columns[0])
        ax1.set_ylabel(X_train.columns[1])
        ax1.set_title("Train Data")
        ax1.legend()

        # Plot test dataset
        colors_test = ['red' if label == 1 else 'blue' for label in Y_test.iloc[:, 0].values]
        ax2.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=colors_test, s=100)
        ax2.scatter([], [], c='red', label='Class 1: Positive')
        ax2.scatter([], [], c='blue', label='Class 2: Negative')
        
        x_min_test, x_max_test = X_test.iloc[:, 0].min(), X_test.iloc[:, 0].max()
        if abs(self.weights[1]) > 1e-6:
            x1_values = np.linspace(x_min_test, x_max_test, 100)
            x2_values = -(self.weights[0] * x1_values + self.bias) / self.weights[1]
            ax2.plot(x1_values, x2_values, color="green", label="Decision Boundary")

        ax2.set_xlabel(X_test.columns[0])
        ax2.set_ylabel(X_test.columns[1])
        ax2.set_title("Test Data")
        ax2.legend()

        # Embed the Matplotlib figure in the Tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=Plot_frame)
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