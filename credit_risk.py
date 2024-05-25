# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:49:47 2024

@author: nirmit27
"""

import os

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency as chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif


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


    
    # Dividing the features into categorical and numerical features seperately
    cat_feats: list[str] = [col for col in df.columns if df[col].dtype == "object"][:-1]
    num_feats: list[str] = [col for col in df.columns if df[col].dtype != "object"][1:]


       
    # Outputting the merged dataset
    df.to_excel(f"{os.path.dirname(os.getcwd())}\\Datasets\\case_study_merged.xlsx")


    
    # Inputting the dataset
    df = pd.read_excel(paths[2], index_col=0)


    
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
    
    
    