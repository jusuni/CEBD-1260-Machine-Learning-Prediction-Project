import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import labelling as lb


def splitting():

    # RUNNING LABELLING
    df_cleaned = lb.labelling()

    # SELECTION OF FEATURES THAT WILL BE USED FOR "X"
    X_features = [f for f in df_cleaned.columns if (df_cleaned[f].dtype not in ["object"]) & (
                f not in ['Unnamed: 0', 'QuoteNumber', 'Original_Quote_Date', 'year', 'month', 'QuoteConversion_Flag'])]

    # CREATING THE DATADSET FOR "X"
    X = df_cleaned[X_features]

    # SELECTING THE TARGET "y" AND CREATING THE DATASET FOR "y"
    y = df_cleaned['QuoteConversion_Flag']

    # SPLITTING INTO TRAIN AND TEST SETS. WE ARE ASIGNING 90% FOR THE TRAINING SET (FOR NOW) AND 10% TO THE
    # TEST SET
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=10, test_size=0.10, stratify=y)

    return df_cleaned, X, y, X_train, y_train, X_test, y_test

