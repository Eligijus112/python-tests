# Importing all the methods for the pipeline 
from venv import create
from pipeline.modules.read_data import read_json
from pipeline.modules.clean_data import clean_text
from pipeline.modules.model_input_preparation import create_X_Y, apply_train_test_split

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

    # Creating the training and testing sets
    train, test = apply_train_test_split(d, test_size=0.2, random_seed=42)

    # Creating the X and Y matrices
    X_train, Y_train = create_X_Y(train, x_col='reviewText', y_col='overall')
    X_test, Y_test = create_X_Y(test, x_col='reviewText', y_col='overall')