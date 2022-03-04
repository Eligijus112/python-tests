# Importing all the methods for the pipeline 
from pipeline.modules.read_data import read_json
from pipeline.modules.clean_data import clean_text
from pipeline.modules.model_input_preparation import create_X_Y, apply_train_test_split
from pipeline.modules.model_fitting import TextCLF
from pipeline.modules.evaluate_model import eval_model

# Directory traversal 
import os 

# Typehinting 
from typing import Tuple

def pipeline(
    input_data_path: str,
    top_features: int = 1000,
    ngram_range: tuple = (1, 1),
) -> Tuple:
    """
    The main pipeline function that reads input data and outputs the model along with statistics on the test set 

    Arguments
    ---------
    input_data_path: str
        The path to the input data
    top_features: int
        The number of top features based on frequency to be used in the model
    ngram_range: tuple
        The ngram range to be used in the model

    Returns
    -------
    clf: TextCLF
        The model that has been fitted to the data
    stats: pd.DataFrame
        The statistics on the test set
    precision: float
        The overall precision of the model
    """
    # Reading the data 
    d = read_json(input_data_path)

    # Cleaning the reviewText column 
    d['reviewText'] = [clean_text(x) for x in d['reviewText']]

    # Creating the training and testing sets
    train, test = apply_train_test_split(d, test_size=0.2, random_seed=42)

    # Creating the X and Y matrices
    X_train, Y_train = create_X_Y(train, x_col='reviewText', y_col='overall')
    X_test, Y_test = create_X_Y(test, x_col='reviewText', y_col='overall')

    # Creating the text classifier object 
    clf = TextCLF(X_train, Y_train)

    # Creating the BOW matrix 
    clf.fit_count_vectorizer(
        top_features=top_features,
        ngram_range=ngram_range
    )

    # Fitting the model
    clf.fit_model()

    # Transforming the X_test into the BOW matrix
    X_test = clf.bow.transform(X_test)

    # Predicting the results
    y_pred = clf.model.predict(X_test)

    # Evaluating the model
    stats, precision = eval_model(Y_test, y_pred)

    # Returning the model object and the stats
    return clf, stats, precision

if __name__ == "__main__":
    # Defining the current file dir 
    _file_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Defining the path to data 
    _data_path = os.path.join(_file_dir, 'data', 'data.json')

    # Applying the pipeline
    clf, stats, precision = pipeline(_data_path)

    # Printing the results
    print(f"Per label statistics:\n{stats}")

    # Calculating the weighted overall average
    print(f"\nOverall weighted precision:\n{round(precision, 3)}")