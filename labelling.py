import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import preprocessing as pp

def labelling():

    # RUNNING PREPROCESSING
    df = pp.preprocessing()

    # CONVERTING "Field6" INTO ONE HOT ENCODING
    add_columns = pd.get_dummies(df["Field6"], prefix="is_Field6")

    # A WAY TO JOIN DATAFRAMES (USING "df.join(df2)")
    df = df.join(add_columns)

    # CONVERTING "Field12" INTO LABEL ENCODER
    df["Field12_label"] = LabelEncoder().fit_transform(df["Field12"])

    # SELECTING THE REMAINING CATEGORICAL COLUMNS TO APPLY ONE HOT ENCODING ALL AT ONCE
    cat_cols = [f for f in df.select_dtypes(include=['object']).columns if
                f not in ["Original_Quote_Date", "Field6", "Field12"]]

    # ONE HOT ENCODING ALL THE REMAININ CATEGORICAL FEATURES. THE SPARSING IS INCLUDED TOR REDUCY MEMORY USAGE
    hot_enc_all_cat_cols_sparsed = pd.get_dummies(df[cat_cols], sparse=True)

    # RENAMING SOME COLUMNS TO AVOID FUTURE ERROR WHEN TRAINING MODEL (REPLACING THE COMMA)
    hot_enc_all_cat_cols_sparsed.rename(columns={"Field10_1,113": "Field10_1_113", "Field10_1,165": "Field10_1_165",
                                                 "Field10_1,480": "Field10_1_480", "Field10_1,487": "Field10_1_487"},
                                        inplace=True)

    # JOING ALL THE ONE HOT ENCODING IN A NEW DATAFRAME NAMED df_hot_sparsed
    df_hot_sparsed = df.join(hot_enc_all_cat_cols_sparsed)

    return df_hot_sparsed