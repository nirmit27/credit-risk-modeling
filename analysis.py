# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:49:47 2024

@author: nirmit27
"""

import os

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import f_oneway as anova
from scipy.stats import chi2_contingency as chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import LabelEncoder as LE

from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV as GSCV

from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from xgboost import XGBClassifier as xgb

from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_recall_fscore_support as prf_support


def make_pred(model, xtrain, xtest, ytrain, ytest):
    model.fit(xtrain, ytrain)

    ypred = model.predict(xtest)

    acc = accuracy(ytest, ypred) * 100
    precision, recall, f1_score, _ = prf_support(ytest, ypred)

    print(f"\nAccuracy = {acc:.2f}%\n")
    for i, v in enumerate(y.unique()):
        print(
            f"Class {v}:\n\tPrecision = {precision[i]:.2f}\n\tRecall = {recall[i]:.2f}\n\tF1-score = {f1_score[i]:.2f}")  # type: ignore


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


if __name__ == "__main__":
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
    df = pd.merge(df1, df2, how="inner",
                  left_on="PROSPECTID", right_on="PROSPECTID")

    df.to_excel(
        f"{os.path.dirname(os.getcwd())}\\datasets\\case_study_merged.xlsx")
    df = pd.read_excel(paths[2], index_col=0)

    """ FEATURE SELECTION """

    # Dividing the features into categorical and numerical features seperately
    cat_feats: list[str] = [
        col for col in df.columns if df[col].dtype == "object"][:-1]
    num_feats: list[str] = [
        col for col in df.columns if df[col].dtype != "object"][1:]

    # Identifying the association between categorical features and target using Contingency tables
    for cat_col in cat_feats:
        c2_score, pval, _, _ = chi2(
            pd.crosstab(df[cat_col], df["Approved_Flag"]))
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


    """ PREPROCESSING """

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

    df["EDUCATION"] = df["EDUCATION"].astype(int)
    cat_feats.pop(1)

    # One Hot Encoding for the remaining nominal categorical features
    df_encoded = pd.get_dummies(df, columns=cat_feats, dtype=int)
    df_encoded.to_excel(
        f"{os.path.dirname(os.getcwd())}\\datasets\\case_study_final.xlsx")

    df_encoded = pd.read_excel(paths[0], index_col=0)

    """" Model Fitting """

    X = df_encoded.drop(columns=['Approved_Flag'])
    y = df_encoded['Approved_Flag']

    label_encoder = LE()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = tts(
        X, y_encoded, test_size=0.2, random_state=27)

    # Decision Tree Classifier

    model0 = DTC(max_depth=20, min_samples_split=10)
    make_pred(model0, X_train, X_test, y_train, y_test)

    # Random Forest

    model1 = RFC(n_estimators=200, random_state=40)
    make_pred(model1, X_train, X_test, y_train, y_test)

    # XGBoost

    model2 = xgb(objective='multi:softmax', num_classes=y.nunique())
    make_pred(model2, X_train, X_test, y_train, y_test)

    # We can observe that P3 class' predictions are very inaccurate.

    """
    Model Accuracies (%) :
        DTC     - 72
        RFC     - 77
        XGBoost - 78
    """

    """ HYPERPARAMETER TUNING """

    # Using XGBoost (choosing this model since it gave the best performance so far)

    params_grid = {
        'colsample_by_tree': [0.3, 0.7, 0.9],
        'alpha': [10, 20, 30],
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 1],
    }

    xgbclf = xgb(objective='multi:softmax', num_class=4)

    # Using Grid Search Cross Validation for finding the best params for highest accuracy ...
    grid = GSCV(estimator=xgbclf, param_grid=params_grid,
                cv=3, n_jobs=-1, scoring='accuracy')
    grid.fit(X, y_encoded)

    print("Best hyperparameters :-")
    for param, value in grid.best_params_.items():
        print(f"{param} : {value}")

    print(f"Best score : {(grid.best_score_ * 100):.2f}%")

    # Using the iterative approach ...
    index = 0

    answers_grid = {
        'combination': [],
        'train_accuracy': [],
        'test_accuracy': [],
        'colsample_bytree': [],
        'learning_rate': [],
        'max_depth': [],
        'alpha': [],
        'n_estimators': []
    }

    for csbt in params_grid['colsample_bytree']:
        for lr in params_grid['learning_rate']:
            for md in params_grid['max_depth']:
                for a in params_grid['alpha']:
                    for ne in params_grid['n_estimators']:

                        index += 1

                        model = xgbclf(objective="multi:softmax", num_class=len(np.unique(y_encoded)),
                                       colsample_bytree=csbt, learning_rate=lr, max_depth=md, alpha=a,
                                       n_estimators=ne)

                        model.fit(X_train, y_train)

                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)

                        train_acc = accuracy(y_train, y_pred_train) * 100
                        test_acc = accuracy(y_test, y_pred_test) * 100

                        answers_grid['combination'].append(index)
                        answers_grid['train_accuracy'].append(train_acc)
                        answers_grid['test_accuracy'].append(test_acc)
                        answers_grid['colsample_bytree'].append(csbt)
                        answers_grid['learning_rate'].append(lr)
                        answers_grid['max_depth'].append(md)
                        answers_grid['alpha'].append(a)
                        answers_grid['n_estimators'].append(ne)

    """
    Best hyperparameters :-
        alpha : 10
        colsample_bytree : 0.9
        learning_rate : 1
        max_depth : 3
        n_estimators : 100
        
    Best score : 81.00%
    """
