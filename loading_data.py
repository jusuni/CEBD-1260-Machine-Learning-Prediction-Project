import numpy as np
import pandas as pd

def loading_raw_data():

    # ASKING THE USER TO INDICATE THE FULL NAME OF THE FILE TO READ
    print("Enter the path and name of the file you want to read")
    print("Format Example: C:\\\\Users\\\\julio\Desktop\\\\Machine Learning\\\\example.csv \n")
    raw_data = input("Path: ")


    # READING THE FILE
    df_raw = pd.read_csv(raw_data)

    return df_raw



