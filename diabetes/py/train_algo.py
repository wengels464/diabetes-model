"""
Start with EDA here. Take a look for:
    Missing values
    Outliers
    Incorrect (high probability data entry error) values
    Correlations which may lead to new features
        Visualize with a correlation matrix
"""

"""
Next we need to:
    Impute missing values using KNN cluster analysis
    Design and add new features
"""

"""
Then prepare the model for training by:
    Using StandardScaler on all the variable fields
    Splitting the data into a stratified 70/30 training/test split.
"""

"""
Fun time. Time to train everything:
    Linear Model
    Logistic Model
    Random Forest
    Naive Bayes
    K Nearest Neighbors
    XGBoost
    Deep Neural Network (maybe not, if feature engineering introduces derivative features)

Hyperparameter Grid Search

Here's an idea:
    Suppose you have a large dataset and are interested in identifying the best algo but don't
    want to churn through AWS bucks. Take a subset of the data that is representative of the dataset
    as a whole and do your brute force search on that. This won't give you the HPs but it will likely
    give you the algo. Then train your algo on the large dataset. I'm sure this has a name...
"""

"""
Okay you've arrived at your optimized model. Now ensure that you haven't overfit by:
    Mapping performance on train v. test data on iterative algos to find a decoupling point:
        Random Forest
        Neural Net (Perceptron, Keras)
    Estimate bias and variance
    Consider boosting or voting?

Validate using k-folds cross validation.
"""

"""
Alright so at this point we have an algorithm that works and has been validated.
Next steps are:
    Pickling and exporting the algo
    Feeding that pickle into the next program
"""

