import pandas as pd
import numpy as np
from SLP import SLP
from Adaline import Adaline
from Preprocessing import preprocessing
import os

feature_index = {'gender':0,'body_mass':1, 'beak_length':2, 'beak_depth':3, 'fin_length':4}
class_index = {'A':0, 'B':1, 'C':2}

def Run(class1,class2, feature1, feature2, learning_rate, n_epochs, model_to_use,bias_bool,TrainFrame,TestFrame):
    print(os.getcwd())
    pass

    all_data = pd.read_csv('Task01/birds.csv')

    f1, f2 = feature_index[feature1], feature_index[feature2]
    b1, b2 = class_index[class1]*50, class_index[class2]*50
    e1, e2 = b1+50, b2+50

    C1 = all_data.iloc[b1:e1,[f1, f2, 5]]
    C2 = all_data.iloc[b2:e2,[f1, f2, 5]]

    # shuffle
    C1 = C1.sample(frac=1).reset_index(drop=True)
    C2 = C2.sample(frac=1).reset_index(drop=True)

    train = pd.concat([C1.iloc[0:30], C2.iloc[0:30]])
    test = pd.concat([C1.iloc[30:], C2.iloc[30:]])

    # shuffle
    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    X_train = train.iloc[:,0:2]
    X_test = test.iloc[:,0:2]

    Y_train = pd.DataFrame(train.iloc[:,2])
    Y_test = pd.DataFrame(test.iloc[:,2])

    if model_to_use == 'SLP':
        my_model = SLP(learning_rate=learning_rate, n_epochs=n_epochs,bias_bool=bias_bool)
    elif model_to_use == 'Adaline':
        my_model = Adaline(learning_rate=learning_rate, n_epochs=n_epochs,bias_bool=bias_bool)

    # Training
    preprocessing(data=X_train)
    preprocessing(data=Y_train, classes=[class1, class2])

    my_model.fit(X=X_train, Y=Y_train)

    Y_train_pred = my_model.predict(X=X_train, column_name='bird category')

    my_model.plot_decision_boundary(X=X_train, Y=Y_train,frame=TrainFrame)

    accuracy = my_model.accuracy_score(Y=Y_train, Y_predict=Y_train_pred)
    print(f"train accuracy = {accuracy}")

    confusion_matrix = my_model.confusion_matrix(Y=Y_train, Y_pred=Y_train_pred)
    print(confusion_matrix)

    # Testing
    preprocessing(data=X_test)
    preprocessing(data=Y_test, classes=[class1, class2])

    Y_test_pred = my_model.predict(X=X_test, column_name='bird category')

    my_model.plot_decision_boundary(X=X_test, Y=Y_test,frame=TestFrame)

    accuracy = my_model.accuracy_score(Y=Y_test, Y_predict=Y_test_pred)
    print(f"train accuracy = {accuracy}")

    confusion_matrix = my_model.confusion_matrix(Y=Y_test, Y_pred=Y_test_pred)
    print(confusion_matrix)

    return accuracy ,confusion_matrix
