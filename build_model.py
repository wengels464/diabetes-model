#!/usr/bin/env python

# General README

"""
This module is an all-in-one script for building a predictive model
from the UCI Pima Indians Dataset (no longer available from UCI).

It takes in a CSV and puts a single serialized sklearn classifier in ../models

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
__status__ = "Live - Testing"
__version__ = "0.0.2"

# Beginning

# general imports

import pathlib
from typing import Union
import pickle as p

# remove ConvergenceWarning from IterativeImputer
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# remove XGB future warnings about deprecation in pandas.Int64Index
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# imports

import pandas as pd
import numpy as np
from scipy import stats

# classifiers

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# metrics

from sklearn.metrics import accuracy_score

# preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# data imputation

from sklearn.experimental import enable_iterative_imputer # noqa
from sklearn.impute import IterativeImputer

# regression models for MICE

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge


# local variables

data_path = pathlib.Path('data/raw/diabetes.csv')
ModelClassifier = Union[RandomForestClassifier, 
                        KNeighborsClassifier, 
                        XGBClassifier]
seed = 25


# define functions

def get_data(data_path: pathlib.WindowsPath) -> pd.DataFrame:
    """
    Takes in a path to a CSV file as a pathlib object.
    
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
        
        """
        Takes a dataframe and replaces all np.nan with the median for the col.
        """
        
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
        
        """
        Code attribution:
            The parameters for these imputers were copied from this source:
        
        Title: Python Feature Engineering Cookbook
        Author: Soledad Galli
        ISBN: 9781789806311
        Link: https://www.packtpub.com/product/python-feature-engineering-cookbook/9781789806311
        
        """
        
        imputer_bayes = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=30,
            random_state=0,
            sample_posterior=False)
        
        imputer_knn = IterativeImputer(
            estimator=KNeighborsRegressor(n_neighbors=5),
            max_iter=30,
            random_state=0,
            sample_posterior=False)
        
        imputer_nonLin = IterativeImputer(
            estimator=DecisionTreeRegressor(
                max_features='sqrt', random_state=0),
            max_iter=30,
            random_state=0,
            sample_posterior=False)
        
        imputer_missForest = IterativeImputer(
            estimator=ExtraTreesRegressor(
                n_estimators=10, random_state=0),
            max_iter=30,
            random_state=0,
            sample_posterior=False)
        
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


def engineer_features(dfs, trim: bool, z: float, scale: bool):
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
        than z standard deviations of clearance.
        
        Returns a dataframe.
        """
        return df[(np.abs(stats.zscore(df)) < z).all(axis=1)]
    

    # apply standard scaler for KNN, minimal impact on other classifiers
    def scale_data(df) -> pd.DataFrame:
        """
        Takes a pandas dataframe and applies sklearn's standard scaler.
        
        Returns a scaled dataframe.
        """
        
        outcomes = df['Outcome']
        unlabeled = df.drop('Outcome', axis=1)
        
        scaler = StandardScaler()
        scaler.fit(unlabeled)
        
        unlabeled = pd.DataFrame(data= scaler.transform(unlabeled),
                            columns = unlabeled.columns)
        
        complete = unlabeled.join(outcomes, how='outer')
        return complete
                        
    
    for key in dfs:
        working_frame = dfs[key]
        
        if trim:
            dfs[key] = trim_outliers(working_frame, z)
            continue
        
        if scale:
            dfs[key] = scale_data(working_frame)
        continue
    
    return dfs
        




def build_model(dfs: dict, model_type: str) -> dict:
    
    """ This function takes a pandas DataFrame of engineered features as output
    by engineer_features and returns a trained model object.

    Args:
       df: pandas DataFrame cleaned features as output by engineer_features
       model_type: a string representing the name of a classifier:
           'KNN' = KNeighborsClassifier()
           'RF' = RandomForestClassifier()

    Returns: a dict of trained_models with labels
    """
    

    
    
    # split data and create data_dict
    def split_data(df):
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=seed, stratify=y)
        
        
        return X_train, X_test, y_train, y_test
    

    # train model    
    def train_model(df, model_type):
        
        # function-specific hyperparamaters
        rf_hps = dict(n_estimators=50,
                      random_state=seed)
        
        knn_hps = dict(n_neighbors=5,
                       weights='uniform',
                       algorithm='brute')
        
        xgb_hps = dict(max_depth=7,
                       eta=0.2,
                       verbosity=0,
                       use_label_encoder = False)
                
        X_train, X_test, y_train, y_test = split_data(df)
        
        clf = None
        
        if model_type == 'KNN':
            hyperparams = knn_hps
            clf = KNeighborsClassifier(**hyperparams)
        elif model_type == 'XGB':
            hyperparams = xgb_hps
            clf = XGBClassifier(**hyperparams)
        elif model_type == 'RF':
            hyperparams = rf_hps
            clf = RandomForestClassifier(**hyperparams)
        
        
        
        clf.fit(X_train, y_train)
        
        
        return clf
    
    # establish dict of trained models and respective scores
    scores = dict()
    models = dict()
    
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
    
    # TODO: Return confusion matrix.
        
    """

    Args:
        data_dict (dict): dict containing X_train, X_test, y_train, y_test

    Returns: metrics

    """

    return None

def summary(model_strings: list) -> pd.DataFrame:
    """
    Parameters
    ----------
    model_strings : list
        This is a list of string abbreviations that the function will
        pass to build_model. Currently, it takes:
            'XGB' - XGBoostClassifier
            'RF' - RandomForestClassifier
            'KNN' - KNeighborsClassifier
        The list is currently list('XGB','RF','KNN')
        
    candidate_dfs : dict
        These are dimensionally identical dataframes that have been
        imputed through different MICE techniques. These have
        been generated by process_data() and engineer_features().
        There are five techniques:
            ExtraTrees
            KNN
            Bayes
            DecisionTree
            median

    Returns
    -------
    A pandas dataframe which contains four columns:
        Model: Either XGB, KNN, or RF
        Object: Actual instance of trained model.
        Dataset: Either Bayes, DecisionTree, RF, KNN, or median.
        Score: Simple accuracy score.
    """

    
    for model in model_strings:
        high_score = 0
        impute_set = None
        impute_object = None
        best_classifier = None
        classifier_object = None
        
        if model == 'KNN':
            for score in scores_knn:
                if scores_knn[score] > high_score:
                    high_score = scores_knn[score]
                    classifier_object = models_knn[score]
                    impute_object = datasets[score]
                    impute_set = score
                    best_classifier = model
                else: continue
            pass
        
        if model == 'XGB':
            for score in scores_xgb:
                if scores_xgb[score] > high_score:
                    high_score = scores_xgb[score]
                    classifier_object = models_xgb[score]
                    impute_object = datasets[score]
                    impute_set = score
                    best_classifier = model
                else: continue
            pass
        
        if model == 'RF':
            for score in scores_rf:
                if scores_rf[score] > high_score:
                    high_score = scores_rf[score]
                    classifier_object = models_rf[score]
                    impute_object = datasets[score]
                    impute_set = score
                    best_classifier = model
                else: continue
            pass
        pass
    
        
        summary =         {"Model Name": best_classifier,
                          "Model Object": classifier_object,
                          "Dataset Name": impute_set,
                          "Dataset Object": impute_object,
                          "Accuracy Score": high_score}
        
        return summary
        
    
# __main__ block

if __name__ == '__main__':
    
    model_list = ["KNN", "RF", "XGB"]
    
    print("Getting data...")
    df = get_data(data_path)
    
    print("Processing data...")
    datasets = process_data(df)
    
    print("Engineering data features...")
    datasets = engineer_features(datasets, trim=False, scale=False, z=np.nan)
    
    print("Constructing classifiers...")
    scores_rf, models_rf = build_model(datasets, 'RF')
    scores_knn, models_knn = build_model(datasets, 'KNN')
    scores_xgb, models_xgb = build_model(datasets, 'XGB')
    
    print("Identifying best performer...")
    outcome = summary(model_list)
    optimal_model_object = outcome["Model Object"]
    
    print("Saving model to file...")
    pickle_path = pathlib.Path('models/model.sav')
    p.dump(optimal_model_object, open(pickle_path, 'wb'))
    
    print("Program successful, exiting...")
    pass

# End