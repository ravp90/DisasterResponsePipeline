import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('ETL_Table', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    # The first chart is a bar chart of the aggregated counts for each genre in the training set.
    # the genre counts are computed and the names are created in a list
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # The second chart (custom) is the aggregated counts of each class.
    # col_names is the names of each column and col_counts is the count of the posts classed positively in each column. 
    col_names = df.columns[-36:]
    col_counts = df.sum().values[-36:]
    
    # The third chart (custom) is a heatmap showing the correlation between classes 
    # The correlation is found using the dataframe's own corr() function and converted to an array. 
    # The purpose of this chart is to visualise how often two or more classes co-existed for a given message
    corr = df.drop(['message','genre'], axis=1).corr().values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    
    # The first graph is the default genre count
    # The second graph creates a sum for each class
    # The third graph is a heatmap of the correlation between classes 
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Classes',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=col_names,
                    y=col_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Classification"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    z=corr,
                    x=col_names,
                    y=col_names
                )
            ],

            'layout': {
                'title': 'Correlation of Message Classes',
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()