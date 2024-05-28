# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:49:47 2024

@author: nirmit27
"""

import os

import pandas as pd
# import numpy as np

from scipy.stats import f_oneway as anova
from scipy.stats import chi2_contingency as chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

# from sklearn.compose import ColumnTransformer as CT
# from sklearn.preprocessing import OneHotEncoder as OHE

from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_recall_fscore_support as prf_support



def unique_vals(cols: list[str], df: pd.DataFrame) -> None:
    for col in cols:
        print(df[col].unique())

def count_nulls(df: pd.DataFrame) -> list[str]:
    col_list: list[str] = []
    
    for column in df.columns:
        if df.loc[df[column] == -99999].shape[0] > 10000:
            col_list.append(column)
    
    return col_list

def fetch_dataset(filepath: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_excel(filepath)
    
    return df

def fetch_dataset_paths() -> list[str]:
    paths: list[str] = []
    
    for dirname, _, filenames in os.walk(os.getcwd()):
        for filename in filenames:
            if filename.endswith(".xlsx"):
                paths.append(os.path.join(dirname, filename))
                
    return paths


if __name__=="__main__":   
    # Reading the datasets
    paths: list[str] = fetch_dataset_paths()
    df1: pd.DataFrame = fetch_dataset(paths[0])
    df2: pd.DataFrame = fetch_dataset(paths[1])
    
    
    
    """ PREPROCESSING """
    
    # Dropping the rows with NULL values i.e. -99999 in df1
    df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]
    
    
    
    # Dropping the columns with > 10,000 NULL values i.e. -99999
    cols: list[str] = count_nulls(df2)
    df2.drop(columns=cols, inplace=True)



    # Dropping the rows with NULL values i.e. -99999 in all columns of df2
    for col in df2.columns:
        df2 = df2.loc[df2[col] != -99999]


    
    # Merging df1 and df2 using INNER JOIN so that no NULL values are present in final df
    df = pd.merge(df1, df2, how="inner", left_on="PROSPECTID", right_on="PROSPECTID")
    df.to_excel(f"{os.path.dirname(os.getcwd())}\\Datasets\\case_study_merged.xlsx")
    
    df = pd.read_excel(paths[2], index_col=0)

    

    """ FEATURE ENGINEERING """

    # Dividing the features into categorical and numerical features seperately
    cat_feats: list[str] = [col for col in df.columns if df[col].dtype == "object"][:-1]
    num_feats: list[str] = [col for col in df.columns if df[col].dtype != "object"][1:]


    
    # Identifying the association between categorical features and target using Contingency tables
    for cat_col in cat_feats:
        c2_score, pval, _, _ = chi2(pd.crosstab(df[cat_col], df["Approved_Flag"]))
        print(f"{cat_col}\t->\t{pval}")
        
    # We will accept ALL the 5 categorical features since they all have a p-value < 0.05 
 
    
 
    """
    Computing VIF for elimination of Multi-collinearity
    We are considering the maximum threshold of 6
    For rejecting the numerical features
    """

    col_index = 0
    num_feats_cols_kept = []
    num_feats_data = df[num_feats]
    total_columns_num_feats_data = num_feats_data.shape[1]   
    
    for i in range(0, total_columns_num_feats_data):
        vif_value = vif(num_feats_data, col_index)
        print(col_index, "\t->\t", vif_value)
        
        if vif_value <= 6:
            num_feats_cols_kept.append(num_feats[i])
            col_index += 1
            
        else:
            num_feats_data.drop(columns=[num_feats[i]], inplace=True)
    
    # Now, we are left with 39 columns of numerical features.
    
    
    
    # Computing the association of numerical features with target categories using ANOVA
    num_feats_cols_kept_2 = []
        
    for col in num_feats_cols_kept:
        a = list(df[col])
        b = list(df["Approved_Flag"])
               
        group_P1 = [value for value, group in zip(a, b) if group == "P1"]
        group_P2 = [value for value, group in zip(a, b) if group == "P2"]
        group_P3 = [value for value, group in zip(a, b) if group == "P3"]
        group_P4 = [value for value, group in zip(a, b) if group == "P4"]
        
        f_score, p_value = anova(group_P1, group_P2, group_P3, group_P4)
    
        if p_value <= 0.05:
            num_feats_cols_kept_2.append(col)
    
    # Now, we have engineered 37 numerical features.
    df = df[cat_feats + num_feats_cols_kept_2 + ["Approved_Flag"]]
    
    
    
    """ FEATURE SELECTION """
    
    unique_vals(cat_feats, df)
    
    # Performing ENCODING for categorical data in categorical features
    
    """
    Ordinal Features :- EDUCATION (arbitrary assignment)
    SSC            = 1
    12TH           = 2
    GRADUATE       = 3
    UNDER GRADUATE = 3
    POST-GRADUATE  = 4
    OTHERS         = 1
    PROFESSIONAL   = 3
    """
    
    df.loc[df["EDUCATION"] == "SSC", ["EDUCATION"]] = 1
    df.loc[df["EDUCATION"] == "12TH", ["EDUCATION"]] = 2
    df.loc[df["EDUCATION"] == "GRADUATE", ["EDUCATION"]] = 3
    df.loc[df["EDUCATION"] == "UNDER GRADUATE", ["EDUCATION"]] = 3
    df.loc[df["EDUCATION"] == "POST-GRADUATE", ["EDUCATION"]] = 4
    df.loc[df["EDUCATION"] == "OTHERS", ["EDUCATION"]] = 1
    df.loc[df["EDUCATION"] == "PROFESSIONAL", ["EDUCATION"]] = 3
    
    cat_feats.pop(1)
    
    # One Hot Encoding for the remaining nominal categorical features
    df_encoded = pd.get_dummies(df, columns = cat_feats, dtype=int)
    
    
    
    """ Model Fitting - Random Forest """
    
    y = df_encoded["Approved_Flag"]
    X = df_encoded.drop(columns=["Approved_Flag"])
    
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=27)
    
    model1 = RFC(n_estimators=200, random_state=40)
    model1.fit(X_train, y_train)
    
    y_pred = model1.predict(X_test)
    
    acc = round(accuracy(y_test, y_pred) * 100, 2)
    precision, recall, f_score, _ = prf_support(y_test, y_pred)
    
    print(f"Accuracy = {acc}%\nPrecision = {precision}\nRecall = {recall}\nF-score = {f_score}")
    
    # Precision, Recall and F-scores for individual classes ...
    for i, v in enumerate(y.unique()):
        print(f"Class {v}:\n\tPrecision = {precision[i]}\n\tRecall = {recall[i]}\n\tF-score = {f_score[i]}")
        
    # We can observe that P3 class' predictions are very inaccurate.