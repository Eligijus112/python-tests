# Importing all the methods for the pipeline 
from pipeline.modules.read_data import read_json
from pipeline.modules.clean_data import clean_text
from pipeline.modules.model_input_preparation import create_X_Y, apply_train_test_split
from pipeline.modules.model_fitting import TextCLF
from pipeline.modules.evaluate_model import eval_model

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

    # Creating the text classifier object 
    clf = TextCLF(X_train, Y_train)

    # Creating the BOW matrix 
    clf.fit_count_vectorizer(
        top_features=100,
        ngram_range=(1, 1)
    )

    # Fitting the model
    clf.fit_model()

    # Transforming the X_test into the BOW matrix
    X_test = clf.bow.transform(X_test)

    # Predicting the results
    y_pred = clf.model.predict(X_test)

    # Evaluating the model
    stats, precision = eval_model(Y_test, y_pred)

    # Printing the results
    print(f"Per label statistics:\n{stats}")

    # Calculating the weighted overall average
    print(f"\nOverall weighted precision:\n{round(precision, 3)}")