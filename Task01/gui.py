from tkinter import ttk
import tkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Toplevel
from tkinter import filedialog, messagebox
import pandas as pd

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

        #print(self.selected_classes)

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

def display_confusion_matrix(matrix):
    # Create a new top-level window in Tkinter
    cm_window = Toplevel()
    cm_window.title("Confusion Matrix")

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=ax)
    ax.set_title("Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    # Embed the plot in Tkinter window using a canvas
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=cm_window)
    canvas.draw()
    canvas.get_tk_widget().pack()


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
    

    class1, class2 = selector.selected_classes
    feature1, feature2 = selector.selected_features
    bias = Bias_var.get() == "1"

    print(class1,class2, feature1, feature2, learning_rate, epochs, bias ,model_to_use)
    accuracy , confusion_matrix = Run(class1,class2, feature1, feature2, learning_rate, epochs, model_to_use,bias)

    messagebox.showinfo("Success", f"Accuracy: {accuracy*100}%")
    display_confusion_matrix(confusion_matrix)
    
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


# Creating a separate frame for class buttons to display them in a row
class_buttons = []
class_names = ['A', 'B', 'C']

class_frame = ttk.Frame(frame, padding="5")
class_frame.grid(row=0, column=1, columnspan=5, sticky=(tk.W, tk.E))
frame.columnconfigure(0, weight=1)  # Allow main frame to stretch

# Label for classes
ttk.Label(frame, text="Select Classes:").grid(column=0, row=0, sticky=tk.W)
# Buttons for classes
for i, class_name in enumerate(class_names):
    button = ttk.Button(
        class_frame,
        text=class_name,
        style='TButton',
        command=lambda idx=i: selector.toggle_class(class_names[idx], class_buttons[idx])
    )
    button.grid(row=0, column=i+1, padx=10, pady=10)  # All buttons in a single row within class_frame
    class_buttons.append(button)




# Creating a separate frame for feature buttons and setting display style
feature_buttons = []
feature_names = ['gender', 'body_mass', 'beak_length', 'beak_depth', 'fin_length']

feature_frame = ttk.Frame(frame, padding="5")
feature_frame.grid(row=2, column=1, columnspan=len(feature_names), sticky=(tk.W, tk.E))
frame.columnconfigure(0, weight=1)  # Allow main frame to stretch


# Label for features
ttk.Label(frame, text="Select Features:").grid(column=0, row=2, sticky=tk.W)


for i, feature_name in enumerate(feature_names):
    button = ttk.Button(
        feature_frame,
        text=feature_name,
        style='TButton',
        command=lambda idx=i: selector.toggle_feature(feature_names[idx], feature_buttons[idx])
    )
    button.grid(row=0, column=i+1, padx=10, pady=10)  # All buttons in a single row
    feature_buttons.append(button)

# Learning Rate an Epochs number
input_frame = ttk.Frame(frame, padding="5")
input_frame.grid(row=4, column=0,columnspan=2, sticky=(tk.W, tk.E) , padx=10, pady=10)

ttk.Label(input_frame, text="Learning Rate:").grid(column=0, row=0, sticky=tk.W, padx=10, pady=10)
learning_rate_entry = ttk.Entry(input_frame)
learning_rate_entry.grid(column=1, row=0, padx=10, pady=10)

ttk.Label(input_frame, text="Number of Epochs:").grid(column=0, row=1, sticky=tk.W, padx=10, pady=10)
epochs_entry = ttk.Entry(input_frame)
epochs_entry.grid(column=1, row=1, padx=10, pady=10)

Bias_var = tk.StringVar(value="1")
Bias_radio = ttk.Radiobutton(input_frame, text="With Bais", variable=Bias_var, value="1")
Bias_radio.grid(row=2, column=0, padx=5, pady=5)
No_Bias_radio = ttk.Radiobutton(input_frame, text="Without Bais", variable=Bias_var, value="0")
No_Bias_radio.grid(row=2, column=1, padx=5, pady=5)



# Run Button
model_frame = ttk.Frame(frame, padding="5")
model_frame.grid(row=6, column=0,columnspan=2, sticky=(tk.W, tk.E) , padx=10, pady=10)
ttk.Label(model_frame, text="Model:").grid(column=0, row=0, sticky=tk.W, padx=10, pady=10)

slp_btn = ttk.Button(model_frame, text="Run SLP", command=lambda: on_run('SLP'))
slp_btn.grid(column=1, row=0, padx=10, pady=10)

adaline_btn = ttk.Button(model_frame, text="Run Adaline", command=lambda: on_run('Adaline'))
adaline_btn.grid(column=2, row=0, padx=10, pady=10)

# Define a new style for the frame background color in ttk
style = ttk.Style()
style.configure("FeatureFrame.TFrame", background="#1e1e1e")  # Set to black or any dark color

root.mainloop()
