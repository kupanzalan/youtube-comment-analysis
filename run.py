import sys
import re
import joblib
import nltk
import pandas as pd
import numpy as np
import comments_collector as cc
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def get_comments(video_url, api_key, no_of_comments, order = 'time',
part = 'snippet', max_results = 100):

    yt_service = cc.build_service(api_key)
    video_id = cc.get_id(video_url)
    print('video_iD: ', video_id)
    comments = cc.comments_helper(video_id, api_key, yt_service, no_of_comments)
    return comments


def clear_text(word_list):

    lemmatized_text = []
    cleaned_text = []
    tokens_final = []

    tokenized_content = nltk.word_tokenize(word_list)

    for i in tokenized_content:
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

    api_key = 'AIzaSyAmDwBSf7dFjcu3bznTCupN-wKr1s_4-Ac'
   
    tokens = []
    word_list = []
    no_of_comments = 100
    i = 1
    N = 100
    NUM_DIFF = 75

    matrix = np.loadtxt("./IMDB_embeddings/Already_done/vectors.txt")

    words = open('./IMDB_embeddings/Already_done/words.txt', 'r')
    lines = words.readlines()

    for line in lines:
        word_list.append(line.split(',')[0][2:][:-1])

    for arg in sys.argv[1:]:
        comments = get_comments(arg, api_key, no_of_comments)
        print(f'Done for video: {i}')
        print(arg)
        i += 1

        print('\n')
        # print('comments for video: ', comments)
    
    comment_no = 0
    empty_vectors = 0
    enough_words = 0
    vector_list = []
    sentiments = []
    analyzed_comments = []

    loaded_model = joblib.load("./random_forest.joblib")

    for rows in comments:
        print(rows)

        cleared_text = clear_text(rows)
        print('cleared_text: ', cleared_text)
        print('\n')

        total_words = 0
        known_word_count = 0

        avg_vec = np.zeros(N)

        if cleared_text:

            for word in cleared_text:
                total_words += 1
                if word in word_list:

                    known_word_count += 1
                    avg_vec += matrix[word_list.index(word)]
                    # print('add: ', matrix[word_list.index(word)])
                    
                else:
                    print(word, ' not in list')
            
            print('total_words: ', total_words)
            print('known_word_count: ', known_word_count)

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

                print('Percentage of known words in this comment: ', known_word_count / total_words * 100, '%')
                print('Enough word embeddings to vectorize text :)')
                enough_words += 1
                # print('word_count: ', word_count)
                # print('average vector: ', avg_vec)

                array_sum = np.sum(avg_vec)
                array_has_nan = np.isnan(array_sum)

                if array_has_nan == False:
                    # sentiments.append(int(rows.Sentiment))
                    vector_list.append(avg_vec)
                    analyzed_comments.append(rows)

                else:
                    print('NaN detected at comment no.: ', comment_no)
                    print('Row: ', avg_vec)
                    print('word_count: ', known_word_count)
                    print('Total words: ', total_words)
                    print('\n')
                    print("Cleaned text: ", cleared_text)
                    print('\n')

                    for word in cleared_text:
                        if word in word_list:
                            print(word, 'is in word list')
                        else:
                            print(word, 'not in word list')
                # print('Sentiments: ', sentiments)

        else:
            print('Empty vector, no. ', comment_no)
        comment_no += 1

    # print(vector_list)

    print(enough_words, 'times was enough embeddings out of', no_of_comments)
    print('\n')

    comments_df = pd.DataFrame(vector_list)
    print(comments_df)

    y_predicted = loaded_model.predict(comments_df)

    print(y_predicted)

    print(analyzed_comments)

    print('\n')
    for i, j in zip(analyzed_comments, y_predicted):
        print(i, ':', j)

    count_1 = 0
    count_2 = 0
    count_3 = 0
    for i in y_predicted:
        if i == 1:
            count_1 += 1
        if i == 2:
            count_2 += 1
        if i == 3:
            count_3 += 1

    print('No. of valid comments: ', len(y_predicted))
    print('No. of 1 values: ', count_1, ' percentage: ', count_1 / len(y_predicted))
    print('No. of 2 values: ', count_2, ' percentage: ', count_2 / len(y_predicted))
    print('No. of 3 values: ', count_3, ' percentage: ', count_3 / len(y_predicted))

    # Some examples for running the program:
    # python3 run.py 'https://www.youtube.com/watch?v=LC3rrqPSEw0'
    # python3 run.py 'https://www.youtube.com/watch?v=Y2d2HLdBF88'
    # python3 run.py 'https://www.youtube.com/watch?v=P3Ife6iBdsU'


        
