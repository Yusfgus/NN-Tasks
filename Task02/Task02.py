import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Preprocessing import preprocessing
from Preprocessing import pre_bird_category
from MLP import MLP


def get_train_test():
    all_data = pd.read_csv('Task02/birds.csv')
    all_data.head()

    # preprocessing
    preprocessing(data=all_data, classes=['A', 'B', 'C']) # A->0 , B->1 , C->2

    # Class A Portion
    Class_A = all_data.iloc[0:50, :]
    # Class B Portion
    Class_B = all_data.iloc[50:100, :]
    # Class C Portion
    Class_C = all_data.iloc[100:150, :]

    # Shuffling portions
    Class_A = Class_A.sample(frac=1, random_state=42).reset_index(drop=True)
    Class_B = Class_B.sample(frac=1, random_state=42).reset_index(drop=True)
    Class_C = Class_C.sample(frac=1, random_state=42).reset_index(drop=True)

    # Data Slicing (Training & Test) for each class
    TrainClass_A = Class_A.iloc[0:30, :]
    TestClass_A = Class_A.iloc[30:50, :]

    TrainClass_B = Class_B.iloc[0:30, :]
    TestClass_B = Class_B.iloc[30:50, :]

    TrainClass_C = Class_C.iloc[0:30, :]
    TestClass_C = Class_C.iloc[30:50, :]


    # Compining Train samples of each class
    train = pd.concat([TrainClass_A, TrainClass_B, TrainClass_C])
    # Compining Test samples of each class
    test = pd.concat([TestClass_A, TestClass_B, TestClass_C])

    # Shuffling Train and Test data
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    test = test.sample(frac=1, random_state=42).reset_index(drop=True)

    # split to X and y
    Xtrain = train.iloc[:, :-1]
    ytrain = train.iloc[:, -1]

    Xtest = test.iloc[:, :-1]
    ytest = test.iloc[:, -1]

    return Xtrain, ytrain, Xtest, ytest




