import pandas as pd
import csv
import sys
import re
import numpy
import math
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams

class Words:

    word_frequency = 0
    tf = 0
    idf = 0
    tf_idf_count = 0

    def __init__(self, word_frequency, tf_idf_count):
        self.word_frequency = word_frequency
        self.tf_idf_count = tf_idf_count

    def __str__(self):
        return "(WordFrequency: %s, TfIDFCount: %s)" % (self.word_frequency, self.tf_idf_count)

def write_matrices(df, word_dictionary):

    original_stdout = sys.stdout

    with open('occurence_matrix.txt', 'w') as f:
        sys.stdout = f

        print("SentenceID   |   ", end=" ")
        for word in word_dictionary:
            print('{:<15}'.format(word), end = " ")

        sys.stdout = original_stdout

def calc_word_frequency(word_dictionary, cleaned_text):

    for word in cleaned_text:
        if word in word_dictionary:
            word_dictionary[word].word_frequency += 1
        else:
            word_dictionary[word] = Words(1, 0)

def calc_tf_idf(df, word_dictionary, cleaned_text):

    no_of_total_documents = len(df.index)

    total_num_of_words = len(word_dictionary)

    for word in word_dictionary:
        for rows in df.itertuples():
            if word in rows.Phrase:
                word_dictionary[word].idf += 1

    for word in word_dictionary:
        word_dictionary[word].tf = word_dictionary[word].word_frequency
        word_dictionary[word].idf = math.log10(no_of_total_documents / (1 + word_dictionary[word].idf))
        
    for word in word_dictionary:
        word_dictionary[word].tf_idf_count = word_dictionary[word].tf * word_dictionary[word].idf
    

def clean_text(df, tokens_final, tokens):

    wordCard = 0
    for rows in df.itertuples():
        tokenized_content = nltk.word_tokenize(rows.Phrase)
        tokens += tokenized_content

    for i in tokens:
        if re.search("^[A-Za-z]+$", i):
            tokens_final.append(i.lower())

     # Porter stemming
    for i in tokens_final:
        stemmed_tokens_porter.append(porter.stem(i))

    # Lancaster steming
        # for i in tokens_final:
        #     stemmed_tokens_lancaster.append(lancaster.stem(i))

    # Filtering out stopwords
    stop_words = set(stopwords.words('english'))

    for i in stemmed_tokens_porter:
        if i not in stop_words:
            cleaned_text.append(i)

    wordCard += len(cleaned_text)

    print("No of words: ", wordCard)
    

if __name__ == "__main__":

    # pd.read_csv('train.tsv', header=None, nrows=5)

    word_dictionary = dict()
    porter = PorterStemmer()
    lancaster = LancasterStemmer()
    tokens = []
    tokens_final = []
    stemmed_tokens_porter = []
    stemmed_tokens_lancaster = []
    cleaned_text = []


    tsv_file = open("train.tsv")
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    cols = ["PhraseID", "SentenceID", "Phrase", "Sentiment"]
    list = []

    next(read_tsv)
    index = 1
    for row in read_tsv:
        if row[1] == str(index):
            index += 1
            list.append(row)
        
        if index == 500:
            break

    df = pd.DataFrame(list, columns=cols)

    print(df)

    clean_text(df, tokens_final, tokens)

    print(cleaned_text)

    calc_word_frequency(word_dictionary, cleaned_text)

    calc_tf_idf(df, word_dictionary, cleaned_text)

    print(word_dictionary['flamboy'])

    # write_matrices(df, word_dictionary)

    tsv_file.close()

