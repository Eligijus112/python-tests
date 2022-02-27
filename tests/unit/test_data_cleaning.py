# The main module in question 
from pipeline.modules.clean_data import clean_text

def test_text_cleaning():
    # Arrange
    _text = [
        'Awesome merchandise....! I really love the 142 features it has',
        'This product is ****! I HATE IT!'
    ]

    # Act
    _cleaned_text = [clean_text(x) for x in _text]

    # Assert 
    assert _cleaned_text == [
        'awesome merchandise i really love the features it has',
        'this product is i hate it'
    ]