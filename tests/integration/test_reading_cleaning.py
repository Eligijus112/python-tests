# Main frameworks to test out 
from pipeline.modules.read_data import read_json
from pipeline.modules.clean_data import clean_text

# Directory traversals 
import os 

def test_reading_cleaning():
    # Arrange 
    _path_to_data = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'data',
        'stub_data.json'
    )

    # Act
    d = read_json(_path_to_data)
    d['reviewText'] = [clean_text(x) for x in d['reviewText']]

    # Assert
    assert d.shape == (5, 2)
    assert set(d.columns) == set(['reviewText', 'overall'])
    assert d['reviewText'][0] == 'i m one of those guys who stops and helps people when they need it'
    assert d['overall'][0] == 5.0