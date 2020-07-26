import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    df = messages_df.merge(categories_df, on=['id'])
    return df


def clean_data(df):
    #joining dataframes
    joint_df_categories = df['categories'].str.split(';', expand = True)
    row = joint_df_categories.iloc[0, :].copy()
    col_names = row.apply(lambda x: x[:-2])
    joint_df_categories.columns = col_names
    df_concat = pd.concat([df, joint_df_categories], axis = 1)
    df = df_concat.iloc[:, 5:].apply(lambda x: x.str.split('-', expand=True)[1])
    for column in df:
        df[column] = pd.to_numeric(df[column])
    df_final = df_concat.drop(df_concat.columns[4:], axis=1)
    df = pd.concat([df_final, df], axis = 1) 
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('mess', engine, index=False)
    return


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