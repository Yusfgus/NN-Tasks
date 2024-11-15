from tkinter import ttk
import tkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Toplevel
from tkinter import filedialog, messagebox
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from task01 import Run


# Class and Feature Button Handler
class FeatureSelector:
    def __init__(self):
        self.selected_classes = []
        self.selected_features = []

    def toggle_class(self, class_name, button):
        if class_name in self.selected_classes:
            self.selected_classes.remove(class_name)
            button.configure(style='TButton')  # Reset to default style
        else:
            if len(self.selected_classes) < 2:
                self.selected_classes.append(class_name)
                button.configure(style='Accent.TButton')
            else:
                messagebox.showwarning("Warning", "You can only select two classes. Deselect one to select another.")


    def toggle_feature(self, feature_name, button):
        if feature_name in self.selected_features:
            self.selected_features.remove(feature_name)
            button.configure(style='TButton')  # Reset to default style
        else:
            if len(self.selected_features) < 2:
                self.selected_features.append(feature_name)
                button.configure(style='Accent.TButton')
            else:
                messagebox.showwarning("Warning", "You can only select two features. Deselect one to select another.")
        #print(self.selected_features)
selector = FeatureSelector()

# Global variable to keep track of the confusion matrix window
cm_window = None

def display_confusion_matrix(matrix):
    global cm_window, canvas

    # If a confusion matrix canvas exists, clear it
    if 'cm_window' in globals() and cm_window is not None:
        cm_window.destroy()

    # Define initial figure size
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=ax)
    ax.set_title("Test Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Embed the plot in Tkinter window using a canvas
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # Create a frame to display the plot
    cm_window = ttk.Frame(right_frame)
    cm_window.grid(row=0, column=4, padx=0, pady=0, sticky="nsew")  # Make the frame resizable

    # Create canvas for the figure
    canvas = FigureCanvasTkAgg(fig, master=cm_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)  # Fill and expand to enable resizing

    # Bind resize events to update the plot size
    def resize_plot(event):
        # Adjust the figure size to match the frame size
        fig.set_size_inches(event.width / 100, event.height / 100)  # Adjust scaling as needed
        fig.tight_layout()
        canvas.draw()

    # Bind the resize event to the canvas' parent frame
    cm_window.bind("<Configure>", resize_plot)


# Function to handle the run button
def on_run(model_to_use='SLP'):

    if model_to_use == 'SLP':
        adaline_btn.configure(style='TButton')
        slp_btn.configure(style='Accent.TButton')
    elif model_to_use == 'Adaline':
        slp_btn.configure(style='TButton')
        adaline_btn.configure(style='Accent.TButton')

    if len(selector.selected_classes) != 2:
        messagebox.showerror("Error", "Please select exactly two different classes.")
        return
    
    if len(selector.selected_features) != 2:
        messagebox.showerror("Error", "Please select exactly two features.")
        return
    

    # Get learning rate and epochs
    try:
        learning_rate = float(learning_rate_entry.get())
        epochs = int(epochs_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers for learning rate and epochs.")
        return
    
    try:
        mx_mse = float(max_mse_entry.get())
    except ValueError:
        mx_mse = 0

    class1, class2 = selector.selected_classes
    feature1, feature2 = selector.selected_features
    bias = Bias_var.get() 

    print(class1,class2, feature1, feature2, learning_rate, epochs, bias ,model_to_use)
    accuracy , confusion_matrix = Run(class1,class2, feature1, feature2, learning_rate, epochs, model_to_use,bias,mx_mse,TrainFrame=Train_frame,TestFrame=Test_frame)

    #messagebox.showinfo("Success", f"Accuracy: {accuracy*100}%")
    label_accuracy.config(text=f"Testing Accuracy: {accuracy*100}%")
    display_confusion_matrix(confusion_matrix)
    
# Create the main application window
root = tk.Tk()
root.title("Signal Reader")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{int(screen_width)}x{screen_height}")
root.state('zoomed')

root.tk.call("source", "Task01/azure.tcl")
root.tk.call("set_theme", "dark")

# Creating UI elements
frame = ttk.Frame(root, padding="5")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Create a new frame for the confusion matrix (right side of the main GUI elements)
right_frame = ttk.Frame(root, padding="5")
right_frame.grid(row=0, column=1 ,rowspan=6, padx=2, pady=2, sticky=(tk.N, tk.S, tk.E))

# Frame for class buttons to display them in a row
class_buttons = []
class_names = ['A', 'B', 'C']
class_frame = ttk.Frame(frame, padding="5")
class_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

# Label for classes
ttk.Label(class_frame, text="Select Classes:").grid(column=0, row=0, sticky=tk.W)

# Buttons for classes
for i, class_name in enumerate(class_names):
    button = ttk.Button(
        class_frame,
        text=class_name,
        style='TButton',
        command=lambda idx=i: selector.toggle_class(class_names[idx], class_buttons[idx])
    )
    button.grid(row=0, column=i+1, padx=5, pady=5)  # All buttons in a single row within class_frame
    class_buttons.append(button)

# Frame for feature buttons and setting display style
feature_buttons = []
feature_names = ['gender', 'body_mass', 'beak_length', 'beak_depth', 'fin_length']
feature_frame = ttk.Frame(frame, padding="5")
feature_frame.grid(row=2, column=0, columnspan=len(feature_names), sticky=(tk.W, tk.E))

# Label for features
ttk.Label(feature_frame, text="Select Features:").grid(column=0, row=0, sticky=tk.W)

# Buttons for features
for i, feature_name in enumerate(feature_names):
    button = ttk.Button(
        feature_frame,
        text=feature_name,
        style='TButton',
        command=lambda idx=i: selector.toggle_feature(feature_names[idx], feature_buttons[idx])
    )
    button.grid(row=0, column=i+1, padx=5, pady=5)  # All buttons in a single row
    feature_buttons.append(button)

# Learning Rate and Epochs number
input_frame = ttk.Frame(frame, padding="5")
input_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)

ttk.Label(input_frame, text="Learning Rate:").grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)
learning_rate_entry = ttk.Entry(input_frame)
learning_rate_entry.grid(column=1, row=0, padx=5, pady=5)

ttk.Label(input_frame, text="Max MSE:").grid(column=2, row=0, sticky=tk.W, padx=5, pady=5)
max_mse_entry = ttk.Entry(input_frame)
max_mse_entry.grid(column=3, row=0, padx=5, pady=5)

ttk.Label(input_frame, text="Number of Epochs:").grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)
epochs_entry = ttk.Entry(input_frame)
epochs_entry.grid(column=1, row=1, padx=5, pady=5)

Bias_var = tk.BooleanVar(value=True)
switch_Bias = ttk.Checkbutton(
    input_frame, text="Bias", style="Switch.TCheckbutton", variable=Bias_var
)
switch_Bias.grid(row=1, column=2, padx=5, pady=4)

# Run Button
model_frame = ttk.Frame(frame, padding="5")
model_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)

slp_btn = ttk.Button(model_frame, text="Run SLP", command=lambda: on_run('SLP'))
slp_btn.grid(column=1, row=0, padx=5, pady=5)

adaline_btn = ttk.Button(model_frame, text="Run Adaline", command=lambda: on_run('Adaline'))
adaline_btn.grid(column=2, row=0, padx=5, pady=5)

label_accuracy = ttk.Label(
    model_frame,
    text="Accuracy: ",
    justify="center",
    font=("-size", 15, "-weight", "bold"),
)
label_accuracy.grid(row=0, column=3, pady=5, columnspan=2)


# # Two frames to display plots for train and test data

# Configure the root window row to allow expansion
root.rowconfigure(7, weight=1)

# Set and configure Train_frame and Test_frame with fixed width
screen_width = root.winfo_screenwidth()
fixed_width = screen_width // 2

Train_frame = ttk.Frame(root, borderwidth=2, width=fixed_width)
Train_frame.grid(row=7, column=0, padx=2, pady=2, sticky="nsew")

Test_frame = ttk.Frame(root, borderwidth=2, width=fixed_width)
Test_frame.grid(row=7, column=1, padx=2, pady=2, sticky="nsew")

# Ensure both frames maintain the specified width and don’t resize dynamically
Train_frame.grid_propagate(False)
Test_frame.grid_propagate(False)

# Define a new style for the frame background color in ttk
style = ttk.Style()
style.configure("FeatureFrame.TFrame", background="#1e1e1e")  # Set to black or any dark color

root.mainloop()
