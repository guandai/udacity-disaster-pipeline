import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) ->  pd.DataFrame:
    '''
    loads message and category data
    input:
        messages_filepath: The file path of messages.
        categories_filepath: The file path of categories.
    output:
        df: The combined dataset
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return pd.merge(messages, categories, on='id')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    clean_data
    input:
        df: The combined dataset of messages and categories.
    output:
        df: Cleaned dataset.
    '''
    # get column name
    categories = df.categories.str.split(';', expand=True)
    col_names = categories.loc[0].apply(lambda x: x[:-2]).values.tolist()
    categories.columns = col_names

    # concat the categories back
    cate_with_last_value = categories.applymap(lambda x: int(x[-1]))
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, cate_with_last_value], axis=1)

    # drop up unused data
    df.dropna(subset=col_names, inplace=True)
    df.drop_duplicates(subset='message', inplace=True)
    df.related.replace(2, 0, inplace=True)

    return df

def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """
    export the result to a db
    input:
        df: a cleaned df
        database_filename: the file path of db

    output:
        None
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('message_table', engine, index=False, if_exists='replace')
    engine.dispose()

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

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
