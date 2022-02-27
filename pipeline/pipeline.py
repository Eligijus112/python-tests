# Importing all the methods for the pipeline 
from pipeline.modules.read_data import read_json
from pipeline.modules.clean_data import clean_text

# Directory traversal 
import os 

if __name__ == "__main__":
    # Defining the current file dir 
    _file_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Defining the path to data 
    _data_path = os.path.join(_file_dir, 'data', 'data.json')

    # Reading the data 
    d = read_json(_data_path)

    # Cleaning the reviewText column 
    d['reviewText'] = [clean_text(x) for x in d['reviewText']]