#bookstores
import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy import create_engine
from plotly.graph_objs import Bar

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
df = pd.read_sql_table('CleanedDataTable', engine)

# load vocabulary
vocabulary_counts, vocabulary_words = joblib.load("../models/vocabulary_stats.pkl")

# load category_stats
category_counts,category_names = joblib.load("../models/category_stats.pkl")

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    graphs = [
        {
            'data': [
                Bar(
                    x=list(vocabulary_words),
                    y=list(vocabulary_counts)
                )
            ],
            'layout': {
                'title': 'The 30 words that are random sampled from vocabulary',
                'yaxis': {
                    'title': "Counts"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=list(category_names),
                    y=list(category_counts)
                )
            ],
            'layout': {
                'title': 'The 36 categories that must be classified',
                'yaxis': {
                    'title': "A number of messages"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    
#plotly in JSON
	id = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
	graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
#web page with plotly
    return render_template('master.html', id=id, graphJSON=graphJSON)

@app.route('/go')
def go():
    query = request.args.get('query', '') 
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
