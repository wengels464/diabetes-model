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

# imports

from typing import Tuple
import pandas as pd
import numpy as np
from numpy import std
from numpy import mean
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# local variables

path_to_data: str = 'data/raw/diabetes.csv'


# define functions

def get_data(path_to_data: str) -> pd.DataFrame:
    """
    Takes in a path to a CSV file. Encodes "Outcome" column as bool.
    
    Returns a pandas dataframe
    """
    
    df = pd.read_csv(path_to_data, dtype={'Outcome':bool})
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

    
# testing block

df = get_data(path_to_data)
df = process_data(df)





























