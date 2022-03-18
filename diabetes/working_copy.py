#!/usr/bin/env python

# General README

"""
This module is an all-in-one script for building a predictive model
from the UCI Pima Indians Dataset (no longer available from UCI).

A link to the main page with the original CSV can be found here:
https://www.kaggle.com/uciml/pima-indians-diabetes-database.
"""

# License

"""
MIT License

Copyright (c) 2022 William Engels

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# File attributes

__author__ = "William Engels"
__contact__ = "wengels464@gmail.com"
__copyright__ = "Copyright 2022 William Engels"
__credits__ = ["William Engels", "Christopher Luiz"]
__date__ = "2022/03/01"
__deprecated__ = False
__email__ =  "wengels464@gmail.com"
__license__ = "MIT"
__maintainer__ = "William Engels"
__status__ = "Testing"
__version__ = "0.0.1"

# beginning of production code

# general imports

import pathlib
from typing import Tuple


# imports

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# local variables

data_path = pathlib.Path('data/raw/diabetes.csv')


# define functions

def get_data(data_path: pathlib.WindowsPath) -> pd.DataFrame:
    """
    Takes in a path to a CSV file. Encodes "Outcome" column as bool.
    
    Returns a pandas dataframe
    """
    
    df = pd.read_csv(data_path)
    return df


def process_data(dataframe) -> pd.DataFrame:
    """
    Takes a pandas dataframe. Performs the following operations:
        1. Replaces zeros in choice columns with np.nan
        2. Calculates the mean for those columns.
        3. Imputes the mean over the np.nan values.
        
    Returns a clean pandas dataframe in place.
    """
    
    # local var, EDA showed these cols have invalid zeros
    cols_with_zeros: list = ['Glucose', 'BloodPressure', 
                             'SkinThickness', 'Insulin', 'BMI']
    
    def zeros_to_nans(dataframe, column_list) -> pd.DataFrame:
        """
        Takes a dataframe and a list of columns with bad zeros.
        Replaces those zeros with np.nan.
        Returns the dataframe with np.nan imputed.
        """
        
        dataframe[column_list] = dataframe[column_list].replace({0:np.nan})
        return dataframe
    
        
    def nans_to_means(dataframe) -> pd.DataFrame:
        
        # build a dict with {col1, mean1, col2, mean2...colN, meanN}
        
        columns_means = dict()
        for column in cols_with_zeros:
            columns_means[column] = np.mean(dataframe[column])
            continue
        
        dataframe = dataframe.fillna(value=columns_means)
        return dataframe
    
    df = zeros_to_nans(dataframe, cols_with_zeros)
    df = nans_to_means(df)
    
    
    return df


def engineer_features(df, run_scaler=False, feature_filter=False):
    """ This function takes a pandas DataFrame as output by process_data
    and returns a pandas DataFrame with features ready for modeling.

    Args:
       df: cleanded pandas DataFrame as output by process_data

    Returns: pandas DataFrame with features ready for modeling
    """
    # judicious outlier trimming
    def trim_outliers(df, z: float) -> pd.DataFrame:
        """
        Takes in a dataframe and removes all observations with more
        than 3 standard deviations of clearance.
        
        Returns a dataframe.
        """
        return df[(np.abs(stats.zscore(df)) < z).all(axis=1)]
    
    def filter_feature_importance(df) -> None:
        """
        Placedholder. Consult Chris
        """
        
        return None

    # feature 3 code
    def scale_data(df) -> pd.DataFrame:
        """
        Probably unnecessary for RF, trees. May be necessary if doing 
        anything else, i.e. ensemble, XGBoost. Consult Chris.
        """
        
        # TODO: fix this so that it puts the label column back
        
        scaler = StandardScaler()
        scaler.fit(df)
        return scaler.transform(df)
    
    def add_features(df) -> pd.DataFrame:

        df.loc[:,'N1']=0
        df.loc[(df['Age']<=30) & (df['Glucose']<=120),'N1']=1
        
        
        
        return df


    df = trim_outliers(df, 4) # setting a z-score of 4 for minimal trim
    
    # flow logic for different classifiers, def=False
    
    if run_scaler: 
        df = scale_data(df)
    if feature_filter:
        df = filter_feature_importance(df)

    #df = add_features(df)
    return df




def build_model(df, model_type) -> Tuple[pd.DataFrame, dict]:
    
    # TODO: Docstring says it returns a trained_model but it looks
    # as if it's returning a dataframe "metrics". Get clarification.
    
    """ This function takes a pandas DataFrame of engineered features as output
    by engineer_features and returns a trained model object and metrics.

    Args:
       df: pandas DataFrame cleaned features as output by engineer_features
       model_type: model to fit

    Returns: trained_model
    """
    
    # TODO: memory management and scope potpourri, many questions.
    
    # split data and create data_dict
    def split_data(df):
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=33, stratify=y)
        
        
        return X_train, X_test, y_train, y_test
    
    
    # func-scoped vars (avoid recomputation)
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    # TODO: unclear how model_type interacts
    
    def train_model(df, model_type):
        
        model = RandomForestClassifier(100, random_state=33)
        model.fit(X_train, y_train)
        
        return model

    # run against test set
    
    
    model = train_model(df, 'placeholder')
    yhat = model.predict(X_test)
    
    # call get_metrics
    
    metrics = accuracy_score(y_test, yhat)

    return df, metrics

def get_metrics(data_dict):
    
    # TODO: Ask Chris about what exactly this is supposed to be.
    # TODO: Seems to be mostly for logging/debugging, but unclear.
    # TODO: NVM, think I get it.
    # TODO: NVM, NVM. Still lost.
    
    """

    Args:
        data_dict (dict): dict containing X_train, X_test, y_train, y_test

    Returns: metrics

    """

    return None

    
# testing block

df = get_data(data_path)
df = process_data(df)
df = engineer_features(df)
df, accuracy = build_model(df, 'placeholder')


# scratchpad































