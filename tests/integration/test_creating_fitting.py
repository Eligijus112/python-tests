# Importing the SUTs to test
from pipeline.modules.model_input_preparation import create_X_Y, apply_train_test_split
from pipeline.modules.model_fitting import TextCLF

# Data wrangling
import pandas as pd 

# Array logic
import numpy as np 

def test_creating_data_fitting_model():
    # Arrange
    _d = pd.DataFrame(
        {
            'reviewText': [
                'this is a review',
                'this is another review',
                'this is a third review',
                'wow a fourth review'
            ], 
            'overall': [
                '1',
                '2',
                '3',
                '4'
            ]
        }
    )

    # Act
    _d_train, _d_test = apply_train_test_split(_d, test_size=0.25, random_seed=42)
    X, Y = create_X_Y(_d_train, x_col='reviewText', y_col='overall')

    clf = TextCLF(X, Y)
    clf.fit_count_vectorizer()
    clf.fit_model()

    ybow = clf.bow.transform(_d_test['reviewText'])
    yhat = clf.model.predict(ybow)
    yhat_proba = clf.model.predict_proba(ybow)

    # Assert 
    assert len(yhat) == 1
    assert np.all([isinstance(y, float) for y in yhat_proba[0]])
    assert len(yhat_proba[0]) == 3