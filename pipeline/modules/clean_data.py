# Regex
import re 

def clean_text(x: str) -> str:
    """
    Applies the text cleaning function for the Amazon reviews

    Arguments
    ---------
    x: str
        Text from amazon review

    Returns
    -------
    str
        Cleaned text 
    """
    # Leaving only the english letters
    x = re.sub('[^a-zA-Z]', ' ', x)

    # Stripping double and more spaces
    x = re.sub(' +', ' ', x)

    # Lowering the text 
    x = x.lower()

    # Stripping the trailing nad leading spaces
    x = x.strip()

    # Returning the text 
    return x