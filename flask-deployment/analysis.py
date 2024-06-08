# -*- coding: utf-8 -*-
"""
Created on Thu June 6 07:21:11 2024

@author: nirmit27
"""

import os
import pickle

import numpy as np
import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

from sklearn.preprocessing import StandardScaler as SS

from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV as GSCV

from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse



def vif_filter(x: pd.DataFrame, limit: int = 5) -> list[str]:
    index: int = 0
    kept_cols: list[str] = []

    data: pd.DataFrame = x.copy()
    cols: pd.Index[str] = x.columns

    for i in range(x.shape[1]):
        vif_val = vif(data, index)
        print(f"{index}\t->\t{vif_val}")

        if vif_val <= limit:
            kept_cols.append(cols[i])
            index += 1
        else:
            data.drop(columns=cols[i], inplace=True)
        
    return kept_cols



def rem_outliers(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        q3 = np.percentile(df[col], 75)
        q1 = np.percentile(df[col], 25)
        
        iqr = q3 - q1
            
        lb = q1 - (5 * iqr)
        ub = q3 + (5 * iqr)
        
        df = df.loc[(df[col] >= lb) & (df[col] <= ub)]
    
    return df
        


def fetch_paths() -> list[str]:
    paths: list[str] = []
    
    for dirname, _, filenames in os.walk(os.getcwd()):
        for filename in filenames:
            if filename.endswith('.xlsx'):
                paths.append(os.path.join(dirname, filename))
    
    return paths



if __name__ == "__main__":
    paths: list[str] = fetch_paths()
    
    df: pd.DataFrame = pd.read_excel(paths[1])
    
    print(df.info())
    
    # Removing unnecessary columns
    df.drop(columns = ['Bank name', 'Year'], inplace=True)
    
    # Handling outliers
    df = rem_outliers(df)
    
    # Data splitting
    X = df.drop(columns=['Basic EPS (Rs.)'])
    y = df['Basic EPS (Rs.)']
    
    # Removing multi-collinearity
    kept_cols: list[str] = vif_filter(x=X, limit=100)
    X = df[kept_cols]
    
    # Standard scaling
    ss = SS()
    X = ss.fit_transform(X)
    
    # Modelling
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=67)
    
    params = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2, 1.0],
                'max_depth': [3, 5, 8, 10],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.5, 0.7, 0.8]
            }
    
    xgbreg = XGBRegressor(objective="reg:squarederror")
    
    grid = GSCV(xgbreg, param_grid=params, cv=3, n_jobs=-1, verbose=2, scoring='r2')
    
    grid.fit(X_train, y_train)
    
    best_params = grid.best_params_
    best_score = grid.best_score_
    
    
    # Training with the best hyperparameters
    model = XGBRegressor(objective='reg:squarederror',
                         colsample_bytree = 0.7,
                         learning_rate    = 1.0,
                         max_depth        = 8,
                         alpha            = 10.0,
                         n_estimators     = 10)
   

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse(y_test, y_pred))
    stddevdiff = y_test.std() / rmse
    
    print(f"R-square : {r2}\nRMSE : {rmse}\nRMSE/Standard Deviation : {stddevdiff}")
        
    
    """ Results """
    # R-squared           = 0.80
    # RMSE                = 8.40
    # RMSE/Std. Deviation = 2.32
        

    # Pickling the saved model
    filename = "model.pkl"
    pickle.dump(model, open(filename, 'wb'))
    
    # Loading the pickled model
    loaded_model = pickle.load(open(filename, 'rb'))
