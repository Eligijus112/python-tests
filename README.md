# Testing in software development 

This project showcases in practise how tests are written and what are they in general when talking about software development.

All the project is heavily influenced by the great book "`Unit Testing. Principles, Practices and Patterns`" by Vladimir Khorikov. You can find the book [here](https://www.manning.com/books/unit-testing).

All the tests are in the `tests` directory. 

The framework used to run the tests is `pytest`. 

# Python environment 

We will use `virtualenv` for managing virtual environments. To install it, run:

```
# Get the latest version
pip install virtualenv
```

To create an empty Python 3.9 environment which is used in this project run the command: 

```
virtualenv --python 3.9 app_env
```

To activate the virtual environment, run the following command:

```
source app_env/bin/activate
```

All the packages with the fixed versions are in the requirements.txt file. To install the packages, run:

```
pip install -r requirements.txt
```

# Article on testing 

The main article that supplement this code can be accessed [here]()

This article is written by me and covers most of the basic topics in testing that will be needed to write robust software.

# Application in question 

The application is a machine learning pipeline that trains a classifier: 

```
f(text) -> class
```

The text is a customer review, and the class is the overall score of the product. 

The pipeline creates a model where given a text, the ML model predicts the review score of the product (1 - 5 stars).