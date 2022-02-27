# Data wrangling 
import pandas as pd 

# JSON functionalities
import json

# Reading a json in the .gz format 
def read_json(file_path:str) -> pd.DataFrame:
    """
    Reads the provided special JSON file format 
    and outputs a dataframe with 2 columns:
    reviewText and overall

    Arguments
    ---------
    file_path: str
        The path to the JSON file to be read

    Returns
    -------
    df: pd.DataFrame
        A dataframe with 2 columns: reviewText and overall
    """
    # reading the json file line by line 
    reviews = []
    with open(file_path, 'rt') as f:
        for line in f:
            line_dict = json.loads(line)

            # Leaving only the reviewText and overall keys
            line_dict = {k:v for k,v in line_dict.items() if k in ['reviewText', 'overall']}

            # Appending to the master list
            reviews.append(line_dict)

    # Converting the list to a dataframe
    df = pd.DataFrame(reviews)

    # Returning the list of reviews
    return df 