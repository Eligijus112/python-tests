# Importing the main functionalities to test 
from pipeline.modules.model_input_preparation import create_X_Y, apply_train_test_split

# Data wrangling
import pandas as pd

# Arranging a dataframe used in the tests
_d = pd.DataFrame(
    {
        'reviewText': [
            'This is a review',
            'This is another review',
            'This is a third review',
            'Wow, a fourth review!'
        ], 
        'overall': [
            '1',
            '2',
            '3',
            '4'
        ]
    }
)

def test_train_test_spliting():
    # Act 
    _d_train, _d_test = apply_train_test_split(_d, test_size=0.25, random_seed=42)

    # Assert
    assert _d_train.shape == (3, 2)
    assert _d_test.shape == (1, 2)
    assert _d_train.index.intersection(_d_test.index).empty

def test_X_Y_creation():
    # Act 
    X, Y = create_X_Y(_d, x_col='reviewText', y_col='overall')

    # Assert
    assert X.shape == (_d.shape[0], )
    assert Y.shape == (_d.shape[0], )
    assert len(X) == len(Y)
