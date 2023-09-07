import csv
import re
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

def get_new_sentiments(sentiments):
    for i in sentiments:
        if i == 'negative':
            new_sentiments.append(0)
        elif i == 'neutral':
            new_sentiments.append(1)
        else:
            new_sentiments.append(2)
    
    return new_sentiments

def clear_text(word_list, tokens_final):

    lemmatized_text = []
    cleaned_text = []
    user_free_list = []
    airlines = ['americanair', 'usairways', 'jetblue', 'southwestair', 'united', 'virginamerica']

    indices = [i for i, x in enumerate(word_list) if x == "@"]

    for element in word_list:
        if word_list.index(element) - 1 not in indices:
            user_free_list.append(element)

    for i in user_free_list:
        if re.search("^[A-Za-z]+$", i):
            tokens_final.append(i.lower())

    stop_words = set(stopwords.words('english'))

    for i in tokens_final:
        if i not in stop_words and i not in airlines:
            cleaned_text.append(i)

    wnl = WordNetLemmatizer()

    for word in cleaned_text:
        lemmatized_text.append(wnl.lemmatize(word))

    return lemmatized_text


if __name__ == "__main__":

    word_dictionary = dict()
    N_ROWS = 10000
    tokens_final = []
    cleaned_text = []

    df = pd.read_csv("./Data/Twitter/archive/tweets.csv", nrows=N_ROWS)

    print(df)

    df = df.drop(['airline_sentiment_confidence', 'negativereason', 'negativereason_confidence', 'airline', 'airline_sentiment_gold', 'name', 'negativereason_gold', 'retweet_count', 'tweet_coord', 'tweet_created', 'tweet_location', 'user_timezone'], axis=1)

    print(df)

    new_sentiments = []
    sentiments = df['airline_sentiment'].tolist()

    new_sentiments = get_new_sentiments(sentiments)

    df['airline_sentiment'] = new_sentiments

    matrix = np.loadtxt("./Twitter_embeddings/Already_done/vectors.txt")

    words = open('./Twitter_embeddings/Already_done/words.txt', 'r')
    lines = words.readlines()

    word_list = []
    word_list_tuples = []
    
    for line in lines:
        word_list_tuples.append((line.split(',')[0][2:][:-1], int(line.split(',')[1][:-2])))
        word_list.append(line.split(',')[0][2:][:-1])

    N = 100
    NUM_DIFF = 95
    avg_vec = np.zeros(N)
    vector_list = []
    sentiments = []
    empty_vectors = 0
    enough_words = 0

    # print('\n')
    # print("word_list: ", word_list)

    for rows in df.itertuples():

        tokens_final = []
        total_words = 0
        known_word_count = 0
        avg_vec = np.zeros(N)

        cleaned_text = clear_text(nltk.word_tokenize(rows.text), tokens_final)
        # print('\n')
        # print("Cleaned text: ", cleaned_text)
        # print('\n')

        if cleaned_text:

            for word in cleaned_text:

                total_words += 1
                if word in word_list:

                    known_word_count += 1
                    avg_vec += matrix[word_list.index(word)]
                    
                else:
                    print(word, ' not in list')

            # print('Required words vectorized in dict.: ', word_count)
            # print('Total words: ', total_words)

            if known_word_count < 1:
                print('Not enough word embeddings, Nan may occur')
                print('\n')
            elif known_word_count / total_words * 100 < NUM_DIFF:
                # Percentage of known words should be at least 60%
                print('Percentage of known words in this comment: ', known_word_count / total_words * 100, '%')
                print('Not enough word embeddings... Too many unknown words, teach me more :(')
                print('\n')
            else:
                avg_vec = avg_vec / known_word_count
                
                # print('Enough word embeddings to vectorize text :)')
                enough_words += 1

                array_sum = np.sum(avg_vec)
                array_has_nan = np.isnan(array_sum)

                if array_has_nan == False:
                    sentiments.append(rows.airline_sentiment)
                    vector_list.append(avg_vec)

                else:
                    print('NaN detected at row: ', rows.tweet_id + empty_vectors)
                    print('Row: ', avg_vec)
                    print('word_count: ', known_word_count)
                    print('Total words: ', total_words)
                    print('\n')
                    print("Cleaned text: ", cleaned_text)
                    print('\n')

                    for word in cleaned_text:
                        if word in word_list:
                            print(word, 'is in word list')
                        else:
                            print(word, 'not in word list')
        else:
            empty_vectors += 1
            print('Empty vector, no. ', rows.tweet_id)

    # print(vector_list)

    print(enough_words, 'times was enough embeddings out of', len(df))

    print('sentiments: ', np.asarray(sentiments))

    words_df = pd.DataFrame(vector_list)
    words_df['sentiment'] = sentiments
    print('\n')
    print(df)
    print(words_df)

    # words_df['is_train'] = np.random.uniform(0, 1, len(words_df)) <= .75
    # train, test = words_df[words_df['is_train'] == True], words_df[words_df['is_train'] == False]

    X_train, X_test, y_train, y_test = train_test_split(words_df.drop(['sentiment'], axis='columns'), sentiments, test_size=0.2)
    print('X_train: ', X_train)
    print('X_test: ', X_test)

    # print('X_train: ', X_train)
    # print('X_test: ', X_test)

    # print('y_train: ', y_train)
    # print('y_test: ', y_test)

    model = RandomForestClassifier(n_jobs=2, n_estimators=500, criterion='gini', warm_start=True, random_state=50, min_samples_leaf = 50)

    print(model.fit(X_train, y_train))

    joblib.dump(model, "./random_forest_twitter.joblib")

    print(model.score(X_test, y_test))

    y_predicted = model.predict(X_test)

    cm = confusion_matrix(y_test, y_predicted)
    print(cm)

    plt.figure(figsize=(10,7))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()


    
        