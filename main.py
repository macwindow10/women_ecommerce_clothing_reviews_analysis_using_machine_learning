import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

nltk.download("stopwords")
stop_words = stopwords.words("english")
new_stopping_words = stop_words[:len(stop_words) - 36]
new_stopping_words.remove("not")
nltk.download('punkt')


def data_collection():
    print('* * * * * Data Collection * * * * *')

    # load data set
    df = pd.read_csv("womens_Clothing_ecommerce_reviews.csv")
    print(df.head())

    # get number of records in data set and columns
    print(df.shape)
    return df


def pre_processing(df):
    print('* * * * * Pre-Processing * * * * *')

    # data cleaning
    # get data types of columns
    # also check for null values
    print(df.info())

    # As we are going to do a NLP project so,
    # we need only ratings and reviews columns.
    # We will drop rest of the columns!
    df.drop(['Unnamed: 0', 'Clothing ID', 'Age', 'Title', 'Recommended IND', 'Positive Feedback Count', 'Division Name',
             'Department Name', 'Class Name'], axis=1, inplace=True)
    df.columns = ['Review', 'Rating']
    print(df.head())

    # check null values
    print(df.isnull().sum())
    print(df.shape)

    df.dropna(subset=['Review'], inplace=True)
    print(df.shape)

    print(df.duplicated().sum())
    df.drop_duplicates(keep='first')
    print(df.shape)

    print(df['Rating'].value_counts())

    sns.countplot(x=df['Rating'])
    plt.show()
    sns.histplot(df['Rating'], kde=True)
    plt.show()
    return df


def text_pre_processing(df):
    print('* * * * * Text Pre-Processing * * * * *')
    processed = remove_punctuation(str(df['Review']))
    print(processed)
    tokenized_data = token(processed.lower())
    print(tokenized_data)
    refined_text = remove_numbers(tokenized_data)
    print(refined_text)
    stopwords_removed_text = remove_stopwords(refined_text)
    print(stopwords_removed_text)
    return processed


def remove_punctuation(df):
    text = re.sub("n't", 'not', df)
    text = re.sub('[^\w\s]', '', df)
    return text


def token(df):
    tokenized_text = word_tokenize(df)
    return tokenized_text


def remove_numbers(df):
    numberless = [t for t in df if t.isalpha()]
    return numberless


def remove_stopwords(df):
    sw = [w for w in df if w not in new_stopping_words]
    return sw


if __name__ == '__main__':
    df = data_collection()
    df = pre_processing(df)
    df = text_pre_processing(df)
