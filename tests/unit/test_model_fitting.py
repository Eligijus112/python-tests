# Importing the testing class in question 
from pipeline.modules.model_fitting import TextCLF

# Array math 
import numpy as np


def test_model_fitting():
    # Arrange
    _x = [
        'awesome product',
        'meh seen better',
        'worst experience ever'
    ]
    _y = [
        '5',
        '3',
        '1'
    ]
    
    clf = TextCLF(_x, _y)
    
    _x_test = [
        'awesome experience',
        'very bad'
        ]

    # Act
    clf.fit_count_vectorizer()
    clf.fit_model()
    _y_hat = clf.model.predict_proba(clf.bow.transform(_x_test))

    # Asserting that the number of predictions is equal to the number of test samples
    assert len(_y_hat) == len(_x_test)

    # Asseting that each prediction has 3 unique classes (same as unique classes in _y)
    assert np.all([len(y) == 3 for y in _y_hat])