# Array math 
import numpy as np 

# Data wrangling 
import pandas as pd 

# Typehinting 
from typing import Tuple

# Random spliting 
from sklearn.model_selection import train_test_split

def create_X_Y(
    d: pd.DataFrame, 
    x_col: str = 'reviewText',
    y_col: str = 'overall'
    ) -> Tuple:
    """
    Creates X and Y matrices used in ML training

    Arguments
    ---------
    d: pd.DataFrame
        The dataframe to be used
    x_col: str
        The column name to be used as X
    y_col: str
        The column name to be used as Y

    Returns
    -------
    X: np.ndarray
        The X matrix
    Y: np.ndarray
        The Y matrix
    """
    # Creating the X matrix
    X = d[x_col].values

    # Creating the Y matrix
    Y = d[y_col].astype(str).values

    # Returning the X and Y matrices
    return X, Y

def apply_train_test_split(
    d: pd.DataFrame,
    test_size: float = 0.2,
    random_seed: int = 42
    ) -> Tuple:
    """
    Splits the X and Y matrices into training and testing sets

    Arguments
    ---------
    d: pd.DataFrame
        The dataframe to be used
    test_size: float
        The size of the test set
    random_seed: int
        The random seed to be used

    Returns
    -------
    X_train: np.ndarray
        The X training matrix
    X_test: np.ndarray
        The X testing matrix
    Y_train: np.ndarray
        The Y training matrix
    Y_test: np.ndarray
        The Y testing matrix
    """
    # Returning the X and Y matrices
    return train_test_split(d, test_size=test_size, random_state=random_seed)