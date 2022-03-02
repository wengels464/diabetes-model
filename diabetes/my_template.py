#%%

#!/usr/bin/env python

""" This module does x, for y, because of z.


Use it by taking the following steps:
step a
step b
step c

Be aware of x, and watch out for z.


MIT License

Copyright (c) 2021 <DEV NAME>

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

# standard lib imports
import pickle
import sys

# other imports
#import numpy as np
import pandas as pd
#from statsmodels.tsa.arima_model import Arima

# local variables
path_to_data = 'data/raw/diabetes.csv'

def get_data(path_to_data): #figure out how to add return annotation

    """This function takes in a path and uses several 
    built-in parameters to do an early cleaning of the 
    data. Returns a Pandas dataframe."""

    df = pd.read_csv(path_to_data) #add extra params here
    return df

df = get_data(path_to_data)

def zeroes_exist(dataframe) -> bool:
    # Check if zeroes in either flt or int form exist. Prints type of zero and returns boolean.
    if 0 in dataframe.values or 0.0 in dataframe.values:
        if 0 in dataframe.values:
            print("Int-type 0 found in dataset")
            return True
        else:
            print("Float-type 0.0 found in dataset")
            return True
    else:
        return False

def count_zeroes(dataframe, column_header) -> int:
    # Takes in a dataframe and a column name and returns number of zeroes (float and int)
    count = dataframe[column_header].value_counts()[0]
    if type(count) != int:
        print("Fuck")
    else:
        return count

count = df['Age']
empty = count.isnull().values.any()

#%%

def process_data(df):
    """ This function processes a pandas DataFrame as output by get_data and
    returns a pandas DataFrame of cleaned data for analysis.

    Args:
       df: pandas DataFrame as output by get_data

    Returns: processed pandas DataFrame ready for analysis
    """
    # process column headers

    # send to lowercase

    # remove null

    # fix types

    return df


def engineer_features(df):
    """ This function takes a pandas DataFrame as output by process_data
    and returns a pandas DataFrame with features ready for modeling.

    Args:
       df: cleanded pandas DataFrame as output by process_data

    Returns: pandas DataFrame with features ready for modeling
    """
    # feature 1 code

    # feature 2 code

    # feature 3 code

    return df


def get_metrics(data_dict):
    """

    Args:
        data_dict (dict): dict containing X_train, X_test, y_train, y_test

    Returns: metrics

    """

    return metrics


def build_model(df, model_type):
    """ This function takes a pandas DataFrame of engineered features as output
    by engineer_features and returns a trained model object and metrics.

    Args:
       df: pandas DataFrame cleaned features as output by engineer_features
       model_type: model to fit

    Returns: trained_model
    """
    # split data and create data_dict

    # train model

    # run against test set

    # call get_metrics


    return df, metrics


def predict(model, data):
    """ Return model prediction.

    Args:
        model: model object with predict method
        data: data point to predict

    Returns:
        prediction

    """
    return model.predict(data)


def main(args):
    """ execute primary module functionality
    """
    # TODO move these config options ot a separate file and read at runtime
    # TODO add conditional argv parsing for the config options

    # load existing model from pickle
    inference_only = False

    # Retrain the model
    train_model = False

    # Save the model as a pickle for future inference.
    save_model = False

    # Destination filepath for saved model. Used both as a target when saving
    # and when retrieving a saved model
    model_filepath = None


    if train_model:
        # read data and prep data
        df = get_data(path1, path2)
        processed_df = proces_data(df)
        feature_df = engineer_features(processed_df)

        # build model and run metrics
        model, metrics = build_model(df, model_type)
        # add logging or print statement to capture metrics, if desired

        if save_model:
            with open(model_filepath, 'wb') as filepath:
                pickle.dump(model, filepath)

    else:
        with open(model_filepath, 'rb') as filepath:
            model = pickle.load(filepath)


    return predict(model, data)


if __name__ == '__main__':
    #print("i'm working")

    args = sys.argv[1:]
    if len(args) > 1:
        # model training code here
        print("I'm training the model")
        print(f"I'm predicting arg1 (predict file: {args[0]}")
        print(f"file1: {args[1]}")
        print(f"file2: {args[2]}")

    else:
        print("I'm predicting arg1 (predict file: {args[0]}")
    main(args)
