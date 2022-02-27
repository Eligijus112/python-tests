# Main modeling class 
from sklearn.linear_model import LogisticRegression

# Bag of words representation creation 
from sklearn.feature_extraction.text import CountVectorizer

# Array math
import numpy as np 


# Modeling class 
class TextCLF: 
    """
    Class that houses methods to create the text classifier for Amazon reviews
    """
    def __init__(self, x: np.array, y: np.array) -> None:
        """
        Initializes the classifier with the given X and Y matrices
        
        Arguments
        ---------
        x: np.array
            The array woth input text 
        y: np.array
            The array with the target values
        """
        self.x = x
        self.y = y
    
    def fit_count_vectorizer(
        self,
        top_features: int = 1000,
        ngram_range: tuple = (1, 2),
        ):
        """
        Creates the vectorizer and fits on the data
        
        Arguments
        ---------
        top_features: int
            The number of features to keep
        """
        # Creating the bag of words representation 
        self.bow = CountVectorizer(
            max_features=top_features,
            ngram_range=ngram_range
        ).fit(self.x)

        # Transforming the internal X matrix 
        self.x = self.bow.transform(self.x)

    def fit_model(
        self,
        max_iter: int = 1000,
        ):
        """
        Fits the logistic regression multiclass model on the data
        """
        # Fitting the model 
        self.model = LogisticRegression(
            multi_class='multinomial',
            solver='newton-cg',
            max_iter=max_iter
        ).fit(self.x, self.y)
