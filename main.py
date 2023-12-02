import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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
    df.drop(['Unnamed: 0', 'Clothing ID', 'Age', 'Title', 'Positive Feedback Count', 'Division Name',
             'Department Name', 'Class Name'], axis=1, inplace=True)
    df.columns = ['Review', 'Rating', 'Recommended IND']
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
    print(df['Recommended IND'].value_counts())

    sns.countplot(x=df['Rating'])
    plt.show()
    sns.histplot(df['Rating'], kde=True)
    plt.show()
    return df


def text_pre_processing(df):
    # print('* * * * * Text Pre-Processing * * * * *')
    processed = remove_punctuation(str(df))
    # print(processed)
    tokenized_data = token(processed.lower())
    # print(tokenized_data)
    refined_text = remove_numbers(tokenized_data)
    # print(refined_text)
    stopwords_removed_text = remove_stopwords(refined_text)
    # print(stopwords_removed_text)
    # return processed
    return " ".join(stopwords_removed_text)


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


def classification_task(model, X_train_scaled, y_train, X_test_scaled, y_test, predic, model_name):
    perf_df = pd.DataFrame(
        {'Train_Score': model.score(X_train_scaled, y_train), "Test_Score": model.score(X_test_scaled, y_test),
         "Precision_Score": precision_score(y_test, predic), "Recall_Score": recall_score(y_test, predic),
         "F1_Score": f1_score(y_test, predic), "accuracy": accuracy_score(y_test, predic)}, index=[model_name])
    return perf_df


if __name__ == '__main__':
    df = data_collection()
    df = pre_processing(df)
    # df = text_pre_processing(df)
    df["Review"] = df["Review"].apply(text_pre_processing)

    print(df.head())
    # create two different dataframe of majority and minority class
    df_majority = df[(df['Recommended IND'] == 1)]
    df_minority = df[(df['Recommended IND'] == 0)]
    # upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=15539,  # to match majority class
                                     random_state=42)  # reproducible results
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_minority_upsampled, df_majority])

    # train test split
    X = df_upsampled["Review"]
    y = df_upsampled["Recommended IND"]
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, stratify=y, random_state=101)
    print(X_train.shape, X_test.shape)

    # vectorization
    tf_idf_vectorizer = TfidfVectorizer()
    X_train_tf_idf = tf_idf_vectorizer.fit_transform(X_train)
    X_test_tf_idf = tf_idf_vectorizer.transform(X_test)
    print(X_test_tf_idf.toarray().shape)
    print(X_train_tf_idf.toarray().shape)
    print(X_train_tf_idf.toarray())
    pd.DataFrame(X_train_tf_idf.toarray(),
                 columns=tf_idf_vectorizer.get_feature_names_out())

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train_tf_idf, y_train)
    pred_2 = lr.predict(X_test_tf_idf)
    Eval_lr = classification_task(lr, X_train_tf_idf, y_train, X_test_tf_idf, y_test, pred_2, "logisitc regression")
    print(Eval_lr)
    conf = confusion_matrix(y_test, pred_2, normalize="all")
    disp = ConfusionMatrixDisplay(conf).plot(cmap=plt.cm.PuBuGn)
    print(disp)
    plt.show()
    # https://www.kaggle.com/code/haneenhossam/women-clothing-reviews-classification-with-rnn

