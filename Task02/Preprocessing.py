import numpy as np
import pandas as pd

def pre_gender(col):
    # Replace 'male' with 1 and 'female' with 0 in the 'gender' column
    col = col.replace({'male': 1, 'female': 0})

    # replace null values with the mode
    # print(f"Null: #{col.isnull().sum()}")
    mode_value = col.mode()[0]
    col.fillna(mode_value, inplace=True)
    # print(f"Null: #{col.isnull().sum()}")

    return col

def normalize(col):
    min_value = col.min()
    max_value = col.max()
    col = (col - min_value) / (max_value - min_value)
    return col

def outliers(col):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1  # Interquartile Range
    
    # Define the bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Calculate the mean of the column
    mean_value = col.mean()
    
    # Replace outliers with the lower and upper bounds
    col = col.clip(lower=lower_bound, upper=upper_bound)

    return col

def pre_body_mass(col):
    outliers(col)
    col = normalize(col)
    return col

def pre_beak_length(col):
    outliers(col)
    col = normalize(col)
    return col

def pre_beak_depth(col):
    outliers(col)
    col = normalize(col)
    return col

def pre_fin_length(col):
    outliers(col)
    col = normalize(col)
    return col

def pre_bird_category(col, classes):
    if len(classes) == 2:
        col = col.replace({classes[0]: 1, classes[1]: -1})
    elif len(classes) == 3:
        col = col.replace({classes[0]: 0, classes[1]: 1, classes[2]: 2})
    return col

def preprocessing(data, classes=[]):
    for col in data.columns:
        if col == 'gender':
            data[col] = pre_gender(data[col])
        elif col == 'body_mass':
            data[col] = pre_body_mass(data[col])
        elif col == 'beak_length':
            data[col] = pre_beak_length(data[col])
        elif col == 'beak_depth':
            data[col] = pre_beak_depth(data[col])
        elif col == 'fin_length':
            data[col] = pre_fin_length(data[col])
        elif col == 'bird category':
            data[col] = pre_bird_category(data[col], classes)

