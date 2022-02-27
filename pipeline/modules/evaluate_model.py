# Various metrics 
from sklearn.metrics import precision_score

# Array math 
import numpy as np 

# Data wrangling 
import pandas as pd

# Typehinting
from typing import Tuple

def eval_model(y_true: np.array, y_pred: np.array) -> Tuple:
    """
    Creates a dataframe with the accuracy metrics

    Arguments
    ---------
    y_true: np.array
        The ture labels 

    y_pred: np.array
        The predicted labels 

    Returns 
    -------
    A dataframe with the various accuracy metrics
    """
    # Creating the dataframe with the predictions and true labels
    _df = pd.DataFrame(
        {
            'y_true': y_true,
            'y_pred': y_pred
        }
    )

    # Grouping by each y_true label and calculating the metrics
    _precision = _df.groupby('y_true', as_index=False).apply(lambda x: precision_score(x['y_true'], x['y_pred'], average='micro'))
    _support = _df.groupby('y_true', as_index=False).apply(lambda x: x.shape[0])

    # Renaming the columns 
    _precision.columns = ['y_true', 'precision_micro']
    _support.columns = ['y_true', 'support']

    # Merging the stats together 
    _stats = _precision.merge(_support, on='y_true')

    # Calculating the overall precision
    _w_precision = _stats['precision_micro'] * (_stats['support'] / _stats['support'].sum())
    _w_precision = _w_precision.sum()

    # Returning the dataframe 
    return _stats, _w_precision