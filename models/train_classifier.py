import sys
from typing import Tuple, List

import joblib
import nltk
import pandas as pd
from sqlalchemy import create_engine
import string

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

nltk.download('stopwords')

STOPWORDS = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
#https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate/34294022
PUNCTUATION = str.maketrans('', '', string.punctuation)

def load_data(database_filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load X, Y, and categories from DB
    input:
        database_filepath: filepath of the database
    output:
        X: feature
        Y: labels
        categories: List of the category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('message_table', engine)
    engine.dispose()

    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    categories = Y.columns.tolist()

    return X, Y, categories


def tokenize(text: str) -> List[str]:
    """
    implement a tokenizer
    input:
        text: input message to tokenize
    output:
        tokens: cleaned tokens
    """
    #remove punctuation and get tokens
    no_punctuation = text.translate(PUNCTUATION).lower()
    tokens = nltk.word_tokenize(no_punctuation)

    # lemmatize without stop words
    result = [lemmatizer.lemmatize(w) for w in tokens if w not in STOPWORDS]
    return result


def build_model() -> GridSearchCV:
    """
    build GridSearch model
    input:
        None

    output:
        cv: GridSearch model result
    """

    # tfidf, dimensionality reduction, and clf
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state = 0))))
    ])
    parameters = {
        'tfidf__smooth_idf':[True, False],
        'clf__estimator__estimator__C': [1, 2, 5]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='precision_samples', cv = 5)
    return cv

def evaluate_model(model: GridSearchCV, X_test: pd.DataFrame, Y_test: pd.DataFrame, categories: List[str]) -> None:
    """
    print classification results
    input:
        model: scikit-learn fitted model
        X_test: X test data
        Y_test: Y test classifications
        categories: the category names
    output:
        None
    """

    Y_pred = model.predict(X_test)

    # print report
    print(classification_report(Y_test, Y_pred, target_names=categories))


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    """
    dumps the model
    input:
        model: fitted model
        model_filepath: save the model to this path

    output:
        None
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, categories = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, categories)

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
