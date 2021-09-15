import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load message and category data from two CSV files. 
    
    Inputs:
    messages_filepath - the path to the messages CSV file.
    categories_filepath - the path to the categories CSV file.
    
    Returns:
    pandas Dataframe of the merged messages and categories data. 
    """
    # The data is loaded from the provided CSV files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories, on='id')
    
    return df


def clean_data(df):
    """
    Cleans the categories data and drops duplicate entries
    
    Inputs: 
    df - pandas DataFrame of the merged messages and categories data
    
    Returns:
    df - pandas DataFrame of the cleaned data. 
    """
    # The categories are extracted from the dataframe's 'categories' column and split on the semi-colon
    categories = df['categories'].str.split(';',expand=True)
    
    # Select the first row
    row = categories.loc[0]
    # extract the list of column names from this row
    # the item[:-2] gets the name portion of the entry
    # discarding the classification value
    category_colnames = [item[:-2] for item in row]
    
    # re-assigning the caterogries column names 
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [categories.iloc[i][column].strip()[-1] for i in range(len(categories))]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the existing categories column
    df = df.drop(['categories'],axis = 1)
    
    # concatenate the categories dataframe to the existing df 
    df = pd.concat([df,categories], axis=1, sort=False)
    
    # the related field contains some values of 2, to make this binary, map the 2s to 1s
    df['related'] = df['related'].map({0:0,1:1,2:1})
    
    # drop any duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Save the dataframe to a SQLite database
    
    Inputs:
    df - pandas DataFrame of the cleaned and merged data.
    database_filename - the name of the database.db file. 
    
    Returns:
    None
    """
    # Save the data to the sqlite database
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('ETL_Table', engine, index=False, if_exists='replace')
    pass


def main():
    """
    Main program. Loads, cleans and saves the data. 
    """
    if len(sys.argv) == 4:

        # initialise the input variables 
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        # load the data 
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # clean the data
        print('Cleaning data...')
        df = clean_data(df)
        
        # save the data
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()