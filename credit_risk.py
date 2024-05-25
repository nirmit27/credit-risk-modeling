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
    paths: list[str] = fetch_dataset_paths()
    
    # Reading the datasets
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
    cat_feats: list[str] = [col for col in df.columns if df[col].dtype == "object"]
    num_feats: list[str] = [col for col in df.columns if df[col].dtype != "object"]
    
    # Determining the association between MARITALSTATUS and Approved_Flag
    
    
    # Outputting the merged dataset
    df.to_excel(f"{os.path.dirname(os.getcwd())}\\Datasets\\case_study_merged.xlsx")
    
    