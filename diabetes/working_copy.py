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
from typing import Union
import pickle as p


# imports

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge



# local variables

data_path = pathlib.Path('data/raw/diabetes.csv')
ModelClassifier = Union[RandomForestClassifier, KNeighborsClassifier]
seed = 25

# hyperparameter dictionaries

rf_hyperparams = dict(n_estimators=100, max_depth=15, 
                      max_features=8, random_state=seed)

knn_hyperparams = dict(n_neighbors=7, weights='distance',
                       algorithm='brute')

# map classifiers to hyperparameters

rf_example = RandomForestClassifier()
rf_type = type(rf_example)

knn_type = type(KNeighborsClassifier())

rf_key = 0

clf_hp = dict(rf_key = rf_hyperparams)


# define functions

def get_data(data_path: pathlib.WindowsPath) -> pd.DataFrame:
    """
    Takes in a path to a CSV file. Encodes "Outcome" column as bool.
    
    Returns a pandas dataframe
    """
    
    df = pd.read_csv(data_path)
    return df


def process_data(dataframe) -> dict:
    """
    Takes a pandas dataframe. Performs the following operations:
        1. Replaces zeros in choice columns with np.nan
        2. Calculates the mean for those columns.
        3. Imputes using mice_impute
        4. Returns a dictionary of strings mapping to dataframes.
            e.g. dict("KNN",pd.DataFrame, etc.)
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
    
        
    def nans_to_medians(dataframe) -> pd.DataFrame:
        
        # build a dict with {col1, mean1, col2, mean2...colN, meanN}
        
        columns_medians = dict()
        for column in cols_with_zeros:
            columns_medians[column] = np.nanmedian(dataframe[column])
            continue
        
        dataframe = dataframe.fillna(value=columns_medians)
        return dataframe
    
    def build_imputers() -> dict:
        """
        Builds a collection of candidate imputers.
        Returns a dict of strings mapping to pre-built imputers.
        """ 
        
        imputer_bayes = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=30,
            random_state=0)
        
        imputer_knn = IterativeImputer(
            estimator=KNeighborsRegressor(n_neighbors=5),
            max_iter=30,
            random_state=0)
        
        imputer_nonLin = IterativeImputer(
            estimator=DecisionTreeRegressor(
                max_features='sqrt', random_state=0),
            max_iter=30,
            random_state=0)
        
        imputer_missForest = IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=10, random_state=0),
            max_iter=30,
            random_state=0)
        
        imputer_dict = dict([('Bayes',imputer_bayes),
                            ('KNN',imputer_knn),
                            ('DecisionTree',imputer_nonLin),
                            ('ExtraTrees',imputer_missForest)])
        
        return imputer_dict
        
    
    def mice_impute(dataframe) -> dict:
        """
        Iteratively imputes every imputer returned by build_imputers.
        
        Returns a dictionary of dataframes with imputations.
        """
        
        imputers = build_imputers()
        imputed_dfs = dict()
        
        for key in imputers:
            temp_copy =  dataframe.copy()
            imputers[key].fit(temp_copy)
            value = pd.DataFrame(data=imputers[key].transform(temp_copy),
                                 columns=dataframe.columns)
            imputed_dfs[key] = value
            continue
        
        return imputed_dfs
            
    nan_df = zeros_to_nans(dataframe, cols_with_zeros)
    candidates = mice_impute(nan_df)
    candidates['median'] = nans_to_medians(nan_df)
    
    return candidates


def engineer_features(dfs):
    """
    Takes a dictionary of dataframes with different imputations.
    Performs the following engineering on the dataframes:
        1. Trims all outliers by a given Z value.
        2. Scales using sklearns StandardScaler
    
    Returns a dictionary of engineered dataframes.
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
        
        outcomes = df['Outcome']
        unlabeled = df.drop('Outcome', axis=1)
        
        scaler = StandardScaler()
        scaler.fit(unlabeled)
        
        unlabeled = pd.DataFrame(data= scaler.transform(unlabeled),
                            columns = unlabeled.columns)
        
        complete = unlabeled.join(outcomes, how='outer')
        return complete
                        
    
    def add_features(df) -> pd.DataFrame:

        df.loc[:,'N1']=0
        df.loc[(df['Age']<=30) & (df['Glucose']<=120),'N1']=1
        
        
        
        return df
    
    for key in dfs:
        working_frame = dfs[key]
        
        dfs[key] = trim_outliers(working_frame, 4) # z of 4
        dfs[key] = scale_data(working_frame)
        continue
    
    return dfs
        




def build_model(dfs: dict, model_type: str) -> dict:
    
    # TODO: Docstring says it returns a trained_model but it looks
    # as if it's returning a dataframe "metrics". Get clarification.
    
    """ This function takes a pandas DataFrame of engineered features as output
    by engineer_features and returns a trained model object and metrics.

    Args:
       df: pandas DataFrame cleaned features as output by engineer_features
       model_type: a string representing the name of a classifier:
           'KNN' = KNeighborsClassifier()
           'RF' = RandomForestClassifier()

    Returns: trained_model
    """
    
    # function-specific hyperparamaters
    rf_hps = dict(n_estimators=50,
                  random_state=seed)
    
    knn_hps = dict(n_neighbors=5,
                   weights='uniform',
                   algorithm='brute')
    
    
    
    # split data and create data_dict
    def split_data(df):
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=seed, stratify=y)
        
        
        return X_train, X_test, y_train, y_test
    
    
    # func-scoped vars (avoid recomputation)

    # train model
    # TODO: unclear how model_type interacts
    
    def train_model(df, model_type):
                
        X_train, X_test, y_train, y_test = split_data(df)
        
        clf = None
        
        if model_type == 'KNN':
            hyperparams = knn_hps
            clf = KNeighborsClassifier(**hyperparams)
        elif model_type == 'RF':
            hyperparams = rf_hps
            clf = RandomForestClassifier(**hyperparams)
        
        
        
        clf.fit(X_train, y_train)
        
        
        return clf
    
    # establish dict of trained models and respective scores
    scores = dict()
    models = dict()

    # run against test set
    
    # iterative loop
    for imputation in dfs:
        
        data = dfs[imputation]
        X_train, X_test, y_train, y_test = split_data(data)
        classifier = train_model(data, model_type)
        yhat = classifier.predict(X_test)
        
        score = accuracy_score(yhat, y_test)
        scores[imputation] = score
        models[imputation] = classifier
        continue
    
    return scores, models
        

        

def get_metrics(data_dict):
    
    # TODO: Ask Chris about what exactly this is supposed to be.
    # TODO: Seems to be mostly for logging/debugging, but unclear.
    # TODO: NVM, think I get it.
    # TODO: NVM, NVM. Still lost.
    
    # Confusion matrix, acc/recall, model evaluation in general
    
    """

    Args:
        data_dict (dict): dict containing X_train, X_test, y_train, y_test

    Returns: metrics

    """

    return None

    
# testing block


df = get_data(data_path)
datasets = process_data(df)
datasets = engineer_features(datasets)


scores_rf, models_rf = build_model(datasets, 'RF')
scores_knn, models_knn = build_model(datasets, 'KNN')

optimal_model = models_rf['DecisionTree']

pickle_path = pathlib.Path('models/model.sav')
p.dump(optimal_model, open(pickle_path, 'wb'))


# scratchpad































