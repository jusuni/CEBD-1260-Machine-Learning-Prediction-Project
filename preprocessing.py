import numpy as np
import pandas as pd
import loading_data as load



def preprocessing():

    # RUNNING LOADING_DATA
    df = load.loading_raw_data()


    # ___PREPROCESSING 1___DEALING_WITH_NULLS
    # DROPPING COLUMNS WITH HIGH PERCENTAGE OF NULLS
    df.drop(['PersonalField84', 'PropertyField29'], axis="columns", inplace=True)

    # SELECTING ONLY THE VARIABLES THAT HAVE NULLS GREATER THATN 0
    null_counts = df.isnull().sum()/df["QuoteNumber"].count()*100
    features = null_counts[null_counts > 0].index

    # FILLING THE NaN VALUES WITH "UNKNOWN"
    for f in features:
        df[f].fillna("UNKNOWN", inplace=True)


    # ___PREPROCESSING 2___REPLACING_BLANK_VLAUE_AND_CHANGING_TYPE
    # REPLACING BLANK VALUE FOR MISSING_VALUE
    df["GeographicField63"].replace(" ", "MISSING_VALUE", inplace=True)

    # CONVERTING THE float64 FEATURES INTO floatXX FEATURES
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')

    # CONVERTING THE int64 FEATURES INTO intXX FEATURES
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')


    # ___PREPROCESSING 3___DROPPING_CORELATED_FEATURES
    # GROUPING THE FEATURES BY TYPE TO DROP THE HIGHLY CORRELATED FEATURES
    Field_col = [col for col in df if col.startswith('Field')]
    CoverageField_col = [col for col in df if col.startswith('CoverageField')]
    SalesField_col = [col for col in df if col.startswith('SalesField')]
    PersonalField_col = [col for col in df if col.startswith('PersonalField')]
    PropertyField_col = [col for col in df if col.startswith('PropertyField')]
    GeographicField_col = [col for col in df if col.startswith('GeographicField')]

    # BUILDING THE CORRELATION MATRIXES AND DROPPING THE CORRELATED FEATURES
    corr_matrix = df[Field_col].corr().abs() #THIS ONE DOES NOT HAVE HIGH CORRELATED FEATURES SO WE SKIP

    # DROPPING HIGHLY CORRELATED FEATURES FOR CoverageField
    corr_matrix = df[CoverageField_col].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    df.drop(to_drop, axis=1, inplace=True)

    # DROPPING HIGHLY CORRELATED FEATURES FOR SalesField
    corr_matrix = df[SalesField_col].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    df.drop(to_drop, axis=1, inplace=True)

    # DROPPING HIGHLY CORRELATED FEATURES FOR PersonalField
    corr_matrix = df[PersonalField_col].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    df.drop(to_drop, axis=1, inplace=True)

    # DROPPING HIGHLY CORRELATED FEATURES FOR PropertyField
    corr_matrix = df[PropertyField_col].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    df.drop(to_drop, axis=1, inplace=True)

    # DROPPING HIGHLY CORRELATED FEATURES FOR GeographicalField
    corr_matrix = df[GeographicField_col].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    df.drop(to_drop, axis=1, inplace=True)


    # ___PREPROCESSING 4___CREATING_NEW_FEATURES
    # CHANGING THE TYPE FORM OBJECT TO DATETIME IN ORDER TO EXTRACT MONTH AND YEAR
    df['Original_Quote_Date'] = pd.to_datetime(df['Original_Quote_Date'])

    # EXTRACTING THE YEAR FROM THE DATE AND SAVING IT IN A NEW COLUMN
    df['year'] = df['Original_Quote_Date'].dt.to_period('Y')

    # EXTRACTING THE MONTH FROM THE DATE AND SAVING IT IN A NEW COLUMN
    df['month'] = pd.DatetimeIndex(df['Original_Quote_Date']).month


    # ___PREPROCESSING 4___CREATING_NEW_FEATURES_WITH_LAMBDA_EXPRESSIONS
    # FUNCTION TO DETERMINE QUARTE
    def to_quarter(x):
        if x == 1 or x == 2 or x == 3:
            return 1
        elif x == 4 or x == 5 or x == 6:
            return 2
        elif x == 7 or x == 8 or x == 9:
            return 3
        elif x == 10 or x == 11 or x == 12:
            return 4
        else:
            return "ERROR"

    # APPLYING THE FUNCTION IN A LAMBDA EXPRESSION
    df["quarter"] = df["month"].apply(to_quarter)


    return df

