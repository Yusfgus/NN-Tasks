# Installations
# pip install customtkinter

# Imports
from tkinter import *
import customtkinter as ctk
from Task02 import get_train_test
from MLP import MLP
from tkinter import filedialog
############################################################
    # Data Read (train, test)
Xtrain, ytrain, Xtest, ytest = get_train_test()

############################################################
    # Global
TrainAccuracy, TestAccuracy = 0, 0
Progress = 0

# GUI
############################################################
    # Main window
root = ctk.CTk()
root.geometry("700x550")
root.title("NN TASKS")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

############################################################
    # Right side frame w620
RightSideFrame = ctk.CTkFrame(master=root, width=670, height=520, border_width=2, border_color="#2a2a2a", fg_color="#282828")
RightSideFrame.place(x=15, rely=0.5, anchor=W)

HeaderLabel = ctk.CTkLabel(master=RightSideFrame, text="TASK II - BACK PROPAGATION ON MULTILAYER NNs", font=("arial", 16, "bold"))
HeaderLabel.place(x=20, y=10, anchor=NW)

############################################################
    # Task content
DashBoardFrame = ctk.CTkFrame(master=RightSideFrame, width=580, height=200, border_width=2, border_color="#333333", fg_color="#2e2e2e")
DashBoardFrame.place(relx=0.5, y= 60, anchor=N)

column_width = 580 // 2.5

# Using grid layout inside DashBoardFrame with 3 columns
DashBoardFrame.grid_columnconfigure(0, weight=1, minsize=column_width)  # Configure 3 columns
DashBoardFrame.grid_columnconfigure(1, weight=1, minsize=column_width)
DashBoardFrame.grid_columnconfigure(2, weight=1, minsize=column_width)

# Adding Widgets in 3 Columns inside DashBoardFrame
NumOfLayers_Label = ctk.CTkLabel(master=DashBoardFrame, text="Num of hidden layers", font=("Arial", 14))
NumOfLayers_Label.grid(row=0, column=0, padx=10, pady=(10, 0))

NumOfEpochs_Label = ctk.CTkLabel(master=DashBoardFrame, text="Num of epochs", font=("Arial", 14))
NumOfEpochs_Label.grid(row=0, column=1, padx=10, pady=(10, 0))

LearningRate_Label = ctk.CTkLabel(master=DashBoardFrame, text="Learning rate", font=("Arial", 14))
LearningRate_Label.grid(row=0, column=2, padx=10, pady=(10, 0))

NumOfNeuronsPerLayer_Label = ctk.CTkLabel(master=DashBoardFrame, text="Num of neurons/layer", font=("Arial", 14))
NumOfNeuronsPerLayer_Label.grid(row=2, column=0, padx=10, pady=(10, 0))

ActivationFunc_Label = ctk.CTkLabel(master=DashBoardFrame, text="Activation function", font=("Arial", 14))
ActivationFunc_Label.grid(row=2, column=1, padx=10, pady=(10, 0))

# Adding a second row of widgets for demonstration
NumOfLayers_Entry = ctk.CTkEntry(master=DashBoardFrame, placeholder_text="Type something...", font=("Arial", 13))
NumOfLayers_Entry.grid(row=1, column=0, padx=10, pady=(0,10))

NumOfEpochs_Entry = ctk.CTkEntry(master=DashBoardFrame, placeholder_text="Type something...", font=("Arial", 13))
NumOfEpochs_Entry.grid(row=1, column=1, padx=10, pady=(0,10))

LearningRate_Entry = ctk.CTkEntry(master=DashBoardFrame, placeholder_text="Type something...", font=("Arial", 13))
LearningRate_Entry.grid(row=1, column=2, padx=10, pady=(0,10))

NumOfNeuronsPerLayer_Entry = ctk.CTkEntry(master=DashBoardFrame, placeholder_text="# Nuerons per layer", font=("Arial", 13))
NumOfNeuronsPerLayer_Entry.grid(row=3, column=0, padx=10, pady=(0,10))

Bias_Entry = ctk.CTkCheckBox(master=DashBoardFrame, text="Enable Bias", font=("Arial", 14))
Bias_Entry.grid(row=3, column=2, padx=10, pady=(0,10))

ActivationFunc_Entry = ctk.CTkComboBox(master=DashBoardFrame,values=["Sigmoid", "Tanh"], font=("Arial", 13))
ActivationFunc_Entry.grid(row=3, column=1, padx=10, pady=(0,10))

# Progress of model
Progress_Label = ctk.CTkLabel(master=RightSideFrame, text=f"Completed: {Progress}%", font=("Arial", 14))
Progress_Label.place(x=20, y=230, anchor=NW)

# Adding labels for Accuracies Produced
Train_Accuracy_Label = ctk.CTkLabel(master=RightSideFrame, text=f"Train Accuracy: {TrainAccuracy}", font=("Arial", 14))
Train_Accuracy_Label.place(x=20, y=270, anchor=NW)

Test_Accuracy_Label = ctk.CTkLabel(master=RightSideFrame, text=f"Test Accuracy: {TestAccuracy}", font=("Arial", 14))
Test_Accuracy_Label.place(x=20, y=300, anchor=NW)

############################################################
    # Functions 

NumOfFeatures = 5
NumOfClasses = 3

def Classify():
    global TrainAccuracy, TestAccuracy,Progress_Label

    # Retrieve and parse user input
    NumOfLayers = int(NumOfLayers_Entry.get())  # Number of layers as integer
    NumOfEpochs = int(NumOfEpochs_Entry.get())  # Number of epochs as integer
    LearningRate = float(LearningRate_Entry.get())  # Learning rate as float
    isBiasEnabled = bool(int(Bias_Entry.get()))  # Convert Bias_Entry to boolean
    ActivationFunc = ActivationFunc_Entry.get().lower()  # Activation function 
    NeuronsPerLayers = [int(x) for x in NumOfNeuronsPerLayer_Entry.get().split(",")]

    MLP_Model = MLP(NeuronsPerLayers, LearningRate, 5, 3, ActivationFunc, NumOfEpochs, isBiasEnabled)
    MLP_Model.fit(Xtrain, ytrain,Progress_Label)
    TrainAccuracy, TestAccuracy, cm_train, cm_test = MLP_Model.calculate_accuracy_and_confusion_matrix(Xtrain, ytrain, Xtest, ytest)

    # Update the labels with the new accuracy values
    Train_Accuracy_Label.configure(text=f"Train Accuracy: {TrainAccuracy:.2f}%")
    Test_Accuracy_Label.configure(text=f"Test Accuracy: {TestAccuracy:.2f}%")

############################################################
    # GUI cont.
def Load_model():
    ActivationFunc = ActivationFunc_Entry.get().lower()  # Activation function 
    layers =[]
    LearningRate =1
    NumOfEpochs=100
    if(ActivationFunc == "sigmoid"):
        layers = [3,4]
        LearningRate = 0.01
        NumOfEpochs = 1000
    else:
        layers = [5]
        LearningRate = 0.001
        NumOfEpochs = 5000
    MLP_Model = MLP(layers, LearningRate, 5, 3, ActivationFunc, NumOfEpochs, True)
    MLP_Model.load()
    TrainAccuracy, TestAccuracy, cm_train, cm_test = MLP_Model.calculate_accuracy_and_confusion_matrix(Xtrain, ytrain, Xtest, ytest)

    # Update the labels with the new accuracy values
    Train_Accuracy_Label.configure(text=f"Train Accuracy: {TrainAccuracy:.2f}%")
    Test_Accuracy_Label.configure(text=f"Test Accuracy: {TestAccuracy:.2f}%")


button = ctk.CTkButton(master = RightSideFrame, text = "Classify Now!", width=150, height=30, text_color="lightgreen", fg_color="darkgreen", 
                       border_width=1, border_color="#008318", command=Classify)
button.place(x=650, y= 500, anchor=SE)

button = ctk.CTkButton(master = RightSideFrame, text = "Load mode To classify!", width=150, height=30, text_color="lightgreen", fg_color="darkgreen", 
                       border_width=1, border_color="#008318", command=Load_model)
button.place(x=500, y= 500, anchor=SE)


############################################################
    # GUI Tail 
root.mainloop()