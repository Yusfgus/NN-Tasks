from tkinter import ttk
import tkinter as tk

from tkinter import filedialog, messagebox
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Perceptron
# from sklearn.metrics import accuracy_score
import pandas as pd



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

selector = FeatureSelector()

# Function to handle the run button
def on_run():
    if len(selector.selected_classes) != 2:
        messagebox.showerror("Error", "Please select exactly two different classes.")
        return
    
    if len(selector.selected_features) != 2:
        messagebox.showerror("Error", "Please select exactly two features.")
        return
    
    class1, class2 = selector.selected_classes
    feature1, feature2 = selector.selected_features

    # Get learning rate and epochs
    try:
        learning_rate = float(learning_rate_entry.get())
        epochs = int(epochs_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers for learning rate and epochs.")
        return
    
# Creating the main application window
root = tk.Tk()
root.title("Signal Reader")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{int(screen_width)}x{screen_height}")
root.state('zoomed')

root.tk.call("source", "Task01/azure.tcl")
root.tk.call("set_theme", "dark")

# Creating UI elements
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
# Classes
ttk.Label(frame, text="Select Classes:").grid(column=0, row=0, sticky=tk.W)
class_buttons = []
class_names = ['A', 'B', 'C']

for i, class_name in enumerate(class_names):
    button = ttk.Button(
        frame,
        text=class_name,
        style='TButton',
        command=lambda idx=i: selector.toggle_class(class_names[idx], class_buttons[idx])
    )
    button.grid(column=i, row=1, padx=5, pady=5)
    class_buttons.append(button)

# Features
ttk.Label(frame, text="Select Features:").grid(column=0, row=2, sticky=tk.W)
feature_buttons = []
feature_names = ['body_mass', 'beak_length', 'beak_depth', 'fin_length']

for i, feature_name in enumerate(feature_names):
    button = ttk.Button(
        frame,
        text=feature_name,
        style='TButton',
        command=lambda idx=i: selector.toggle_feature(feature_names[idx], feature_buttons[idx])
    )
    button.grid(column=i, row=3, padx=5, pady=5)
    feature_buttons.append(button)

# Learning Rate and Epochs
ttk.Label(frame, text="Learning Rate:").grid(column=0, row=4, sticky=tk.W)
learning_rate_entry = ttk.Entry(frame)
learning_rate_entry.grid(column=1, row=4)

ttk.Label(frame, text="Number of Epochs:").grid(column=0, row=5, sticky=tk.W)
epochs_entry = ttk.Entry(frame)
epochs_entry.grid(column=1, row=5)

# Run Button
run_button = ttk.Button(frame, text="Run SLP", command=on_run)
run_button.grid(column=0, row=6, columnspan=2)

# Adding a style for selected buttons
style = ttk.Style()
style.configure('Selected.TButton', background='blue', foreground='white')

root.mainloop()
