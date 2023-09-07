import csv
import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer 
from sklearn.decomposition import PCA
from matplotlib import pyplot

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

    # print('Before lemmatization: ', cleaned_text)

    for word in cleaned_text:
        # print('word: ', word)
        # print('lemmatized word: ', wnl.lemmatize(word))
        lemmatized_text.append(wnl.lemmatize(word))

    return lemmatized_text

if __name__ == "__main__":

    sentences = []
    no_of_documents = 4000

    tsv_file = open("./Data/train.tsv")
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    cols = ["PhraseID", "SentenceID", "Phrase", "Sentiment"]
    word_list = []

    next(read_tsv)
    index = 1
    for row in read_tsv:
        if row[1] == str(index):
            index += 1
            word_list.append(row)
        
        if index == no_of_documents:
            break

    tsv_file.close()

    df = pd.DataFrame(word_list, columns=cols)

    print(df)

    for rows in df.itertuples():

        tokens_final = []  
        cleaned_text = clear_text(nltk.word_tokenize(rows.Phrase), tokens_final)

        # print("Cleaned text: ", cleaned_text)
        # print('\n')

        sentences.append(cleaned_text)
    
    print(sentences)

    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    print(model)

    words = list(model.wv.key_to_index)
    print(words)

    np.savetxt('./Gensim_embeddings/vectors_gensim.txt', model.wv.get_normed_vectors(), fmt='%.8f')

    words_num = 0
    with open('./Gensim_embeddings/words_gensim.txt', 'w') as fout:
        for word in list(model.wv.key_to_index):
            fout.write(''.join(str([word, words_num]) + '\n'))
            words_num += 1

    # Plotting works better with low number of documents, ex. no_of_documents = 40
    #         normed_vector = model.wv.get_vector(word, norm=True)
    #         print(normed_vector)

    # X = model.wv.get_normed_vectors()

    # pca = PCA(n_components=2)
    # result = pca.fit_transform(X)

    # pyplot.scatter(result[:, 0], result[:, 1])
    # words = list(model.wv.index_to_key)
    # for i, word in enumerate(words):
	#     pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    # pyplot.show()