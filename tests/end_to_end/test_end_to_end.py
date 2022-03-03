# Importing the main pipeline function
from pipeline.pipeline import pipeline

# Directory traversal 
import os 

# Data frame objects 
import pandas as pd

# Defining the test 
def test_pipeline():
    """
    The test to test out the pipeline end-to-end 
    """ 
    # Arrange 
    _path_to_data = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'data',
        'stub_data.json'
    )
    _test_data = [
        'this product is amazing', 
        'this product - not so much',
        'am I a test text?'
    ]

    ## Act
    # Executing the pipeline
    clf, stats, precision = pipeline(_path_to_data)
    
    # Predicting on the test data
    _test_hat = clf.bow.transform(_test_data)
    _test_hat_labels = clf.model.predict(_test_hat)
    _test_hat_proba = clf.model.predict_proba(_test_hat)

    # Assert 
    assert isinstance(precision, float)
    assert isinstance(stats, pd.DataFrame)
    assert stats.columns.tolist() == ['y_true', 'precision_micro', 'support']
    assert len(_test_hat_labels) == 3 # Number of obs in the test set
    assert _test_hat_proba.shape == (3, 5) # Number of obs and number of classes