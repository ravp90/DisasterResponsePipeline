import sys
import pandas as pd
import numpy as np
import pickle
import re
from sqlalchemy import create_engine
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

def load_data(database_filepath):
    """
    Load the data from the SQLite database and create the input X and output Y DataFrames. 
    
    Inputs: 
    database_filepath - the location of the database.db file
    
    Return:
    X - pandas DataFrame of the model inputs 
    Y - pandas DataFrame of the model output classes
    Y.colums - list of the classes
    """
    # Load the data from the sqlite database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('ETL_Table', engine)
    # create the model input X data
    X = df['message']
    # create the model output Y class data
    Y = df.drop(['id','message','original','genre'], axis=1) 
    # the related field contains some values of 2, to make this binary, map the 2s to 1s
    # there are very few of these 2's, so they were re-mapped to 1 assuming they may have been misclassified. 
    # it is expected that the impact on the model will be negligible given the size of the dataset.
    Y['related'] = Y['related'].map({0:0,1:1,2:1})
    return X, Y, Y.columns


def tokenize(text):
    """
    Tokenise the text input data, replacing the URL with a placeholder, removing stop words and lemmatizing the words. 
    
    Inputs:
    text - a text input or message 
    
    Returns:
    tokens - tokenized form of the text input. 
    """
    # regex for URLs to be replaced with a placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex,text)
    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")
    # the words in the text input to then be split, tokenised and lemmatized, removing stop words. 
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words("english")]
    return tokens


def build_model():
    """
    Builds the model pipeline, defines the gridsearch parameters and returns the gridsearch model object
    
    Inputs:
    None
    
    Returns:
    cv - gridsearch model with pipeline and parameters defined. 
    """
    # The model pipeline, where CountVectorizer uses the tokenizer, TF-IDF is applied and multi-output classifier uses AdaBoostClassifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(),n_jobs=-1))
    ])
    
    # The parameters for the gridsearch, applied only to the AdaBoostClassifier
    parameters = {
        'clf__estimator__n_estimators':[10,50],
        'clf__estimator__learning_rate':[0.01,0.05],
    }

    # create the gridsearch pipeline and output it as the model
    cv = GridSearchCV(pipeline,param_grid=parameters, cv=2, verbose=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model with test data and outputs the classification report
    
    Inputs:
    model - the trained model
    X_test - pandas DataFrame of test data 
    Y_test - pandas DataFrame of test classes
    category_names - list of categories (classes)
    
    Returns:
    None
    """
    # test data is used with model to generate predictions
    y_pred = model.predict(X_test)
    
    # predictions output is an array, converted to a dataframe and column names applied
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = category_names

    # the classification report is called for each class to report the precision, recall and f1 score. 
    print(classification_report(Y_test, y_pred, target_names=category_names))
    return 


def save_model(model, model_filepath):
    """
    Saves the model as a pickle file.
    
    Inputs:
    model - the trained model
    model_filepath - the path to the location where the model will be saved.
    
    Returns:
    None
    """
    # model is saved as a pickle file
    pickle.dump(model,open(model_filepath,'wb'))
    return


def main():
    """
    Main program. Loads the data, splits into training and test set. Builds and trains model. Evaluates and saves model. 
    """
    if len(sys.argv) == 3:
        # inputs variables are initialised
        database_filepath, model_filepath = sys.argv[1:]
        
        # The data is loaded and split into training and test sets
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        # The gridsearch model is built
        print('Building model...')
        model = build_model()
        
        # The gridsearch model is trained with the training set
        print('Training model...')
        model.fit(X_train, Y_train)
        
        # The trained model is evaluated with the test set and the classification report is printed
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        # The model is saved as a pickle file ready for use in the app.
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()