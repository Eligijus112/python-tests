# Importing the main function to test 
from pipeline.modules.evaluate_model import eval_model

def test_model_evaluation():
    # Arrange 
    _y_true = [
        '1',
        '2',
        '1',
        '3'
    ]

    _y_pred = [
        '1',
        '2',
        '1',
        '4'
    ]

    # Act
    _stats, _precision = eval_model(_y_true, _y_pred)

    # Asserting
    assert _stats.shape[0] == 3

    assert _precision == 0.75

    assert _stats.loc[_stats['y_true'] == '1', 'precision_micro'].iloc[0] == 1
    assert _stats.loc[_stats['y_true'] == '1', 'support'].iloc[0] == 2
    
    assert _stats.loc[_stats['y_true'] == '2', 'precision_micro'].iloc[0] == 1
    assert _stats.loc[_stats['y_true'] == '2', 'support'].iloc[0] == 1
    
    assert _stats.loc[_stats['y_true'] == '3', 'precision_micro'].iloc[0] == 0
    assert _stats.loc[_stats['y_true'] == '3', 'support'].iloc[0] == 1
