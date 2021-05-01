import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
import joblib
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

# load data and model
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_table', engine)
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Show distribution of different category
    cat_col = list(df.columns[4:])
    cat_col_counts = []
    for column_name in cat_col:
        cat_col_counts.append(np.sum(df[column_name]))
    # sort distrubution 
    dfc = pd.DataFrame({'cname': cat_col, 'count': cat_col_counts})
    dfc = dfc.sort_values(by='count', ascending=False)
    category = dfc['cname']
    category_counts = dfc['count']

    # get message length distribution
    lengths = df.message.str.split().str.len()
    len_counts, len_div = np.histogram(
        lengths,
        range=(0, lengths.quantile(0.99))
    )

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    marker= {'color': '#4444bb'},
                    x=genre_names,
                    y=genre_counts,
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
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
                    marker= {'color': '#44bbaa'},
                    x=category,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    marker= {'color': '#bb44aa'},
                    x=len_div,
                    y=len_counts
                    )
                 ],
            'layout': {
                'title': 'Message Length Distribution (quantile=0.99)',
                
                'yaxis': {
                    'title': "length count"
                },
                'xaxis': {
                    'title': "Message Length dviation"
                }
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
