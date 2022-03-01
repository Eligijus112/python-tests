# Importing the function to test 
from pipeline.modules.read_data import read_json

# Dir traversal
import os 

def test_data_reading():
    """
    Tests the data reading functionality of the pipeline
    """
    # Arrange 
    _path_to_data = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'data',
        'stub_data.json'
    )

    # Act
    d = read_json(_path_to_data) 

    # Assert
    assert d.shape == (3, 2)
    assert set(d.columns) == set(['reviewText', 'overall'])