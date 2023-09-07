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
        if i == '1' or i == '0':
            new_sentiments.append('1')
        elif i == '3' or i == '4':
            new_sentiments.append('3')
        else:
            new_sentiments.append('2')
    
    return new_sentiments

def clear_text(word_list, tokens_final):

    lemmatized_text = []
    cleaned_text = []

    for i in word_list:
        if re.search("^[A-Za-z]+$", i):
            tokens_final.append(i.lower())

    stop_words = set(stopwords.words('english'))

    for i in tokens_final:
        if i not in stop_words:
            cleaned_text.append(i)

    wnl = WordNetLemmatizer()

    for word in cleaned_text:
        lemmatized_text.append(wnl.lemmatize(word))

    return lemmatized_text

if __name__ == "__main__":

    word_dictionary = dict()
    tokens_final = []
    cleaned_text = []

    no_of_documents = 10000

    tsv_file = open("./Data/train.tsv")
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    cols = ["PhraseID", "SentenceID", "Phrase", "Sentiment"]
    word_list = []

    next(read_tsv)
    index = 1

    for row in read_tsv:
        index += 1
        word_list.append(row)
        
        if index == no_of_documents:
            break

    tsv_file.close()

    df = pd.DataFrame(word_list, columns=cols)

    print(df)

    new_sentiments = []
    sentiments = df['Sentiment'].tolist()

    new_sentiments = get_new_sentiments(sentiments)

    df['Sentiment'] = new_sentiments

    matrix = np.loadtxt("./IMDB_embeddings/Already_done/vectors.txt")

    words = open('./IMDB_embeddings/Already_done/words.txt', 'r')
    lines = words.readlines()

    word_list = []
    word_list_tuples = []
    
    for line in lines:
        word_list_tuples.append((line.split(',')[0][2:][:-1], int(line.split(',')[1][:-2])))
        word_list.append(line.split(',')[0][2:][:-1])

    N = 100
    NUM_DIFF = 65
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

        cleaned_text = clear_text(nltk.word_tokenize(rows.Phrase), tokens_final)
        # print('\n')
        # print("Cleaned text: ", cleaned_text)
        # print('\n')

        if cleaned_text:

            for word in cleaned_text:

                total_words += 1
                if word in word_list:

                    known_word_count += 1
                    avg_vec += matrix[word_list.index(word)]

                    # print('add: ', matrix[word_list.index(word)])
                    
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
                # print('average vector before: ', avg_vec)
                
                avg_vec = avg_vec / known_word_count
                
                # print('Enough word embeddings to vectorize text :)')
                enough_words += 1
                # print('word_count: ', word_count)
                # print('average vector: ', avg_vec)

                array_sum = np.sum(avg_vec)
                array_has_nan = np.isnan(array_sum)

                if array_has_nan == False:
                    sentiments.append(int(rows.Sentiment))
                    vector_list.append(avg_vec)

                else:
                    print('NaN detected at row: ', int(rows.SentenceID) - 2 + empty_vectors)
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
                # print('Sentiments: ', sentiments)
        else:
            empty_vectors += 1
            print('Empty vector, no. ', int(rows.SentenceID) - 1)

    # print(vector_list)
    # print(enough_words, 'times was enough embeddings out of', len(df))
    # print('sentiments: ', sentiments)

    words_df = pd.DataFrame(vector_list)
    words_df['sentiment'] = sentiments
    # words_df['is_train'] = np.random.uniform(0, 1, len(df) - empty_vectors) <= .75
    print('\n')
    print(df)
    print(words_df)

    X_train, X_test, y_train, y_test = train_test_split(words_df.drop(['sentiment'], axis='columns'), sentiments, test_size=0.2)

    # print('X_train: ', X_train)
    # print('X_test: ', X_test)

    # print('y_train: ', y_train)
    # print('y_test: ', y_test)
    
    model = RandomForestClassifier(n_jobs=2, n_estimators=100, criterion='gini', random_state=0, max_features=10)

    print(model.fit(X_train, y_train))

    joblib.dump(model, "./random_forest.joblib")

    print(model.score(X_test, y_test))

    y_predicted = model.predict(X_test)

    cm = confusion_matrix(y_test, y_predicted)
    print(cm)

    plt.figure(figsize=(10,7))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()
        