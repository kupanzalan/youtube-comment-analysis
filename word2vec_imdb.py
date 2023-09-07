import csv
import re
import pandas as pd
import numpy as np
import nltk
import gc
import os
from itertools import islice
from time import process_time
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from matplotlib import pyplot
import matplotlib.pyplot as plt

def next_n_lines(file_opened, N):
    return [x.strip() for x in islice(file_opened, N)]

def clear_text(word_list, tokens_final):

    # wordCard = 0
    lemmatized_text = []
    cleaned_text = []

    # for rows in word_list:
    #     tokenized_content = nltk.word_tokenize(rows)
    #     tokens += tokenized_content

    for i in word_list:
        if re.search("^[A-Za-z]+$", i):
            tokens_final.append(i.lower())

    stop_words = set(stopwords.words('english'))

    for i in tokens_final:
        if i not in stop_words:
            cleaned_text.append(i)

    # wordCard += len(cleaned_text)

    wnl = WordNetLemmatizer()

    # print('Before lemmatization: ', cleaned_text)

    for word in cleaned_text:
        # print('word: ', word)
        # print('lemmatized word: ', wnl.lemmatize(word))
        lemmatized_text.append(wnl.lemmatize(word))

    return lemmatized_text

def mapping(cleaned_text):

    word_to_id = dict()
    id_to_word = dict()

    for i, token in enumerate(set(cleaned_text)):
        word_to_id[token] = i
        id_to_word[i] = token

    return word_to_id, id_to_word

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% - Generating training data

def generate_training_data(cleaned_text, word_to_id, id_to_word, window_size, vocab_size):

    n = len(cleaned_text)

    # print("N: ", n)

    input_list, output_list = [], []

    for i in range(n):
        nbr_inds = list(range(max(0, i - window_size), i)) + list(range(i + 1, min(n, i + 1 + window_size)))

        # print("nbr_inds: ", nbr_inds)

        for j in nbr_inds:
            input_list.append(word_to_id[cleaned_text[i]])
            # print("X: ", X)
            output_list.append(word_to_id[cleaned_text[j]])
            # print("Y: ", Y)

    # print("input_list: ", input_list)
    input_list = np.array(input_list)
    input_list = np.expand_dims(input_list, axis=0)

    # print("output_list: ", output_list)
    output_list = np.array(output_list)
    output_list = np.expand_dims(output_list, axis=0)

    # print("input_list: ", input_list)
    # print("output_list: ", output_list)

    # print("Shape of output_list: ", output_list.shape)
    m = output_list.shape[1]
    # print("m = output_list.shape[1]: ", m)

    input_list, output_list = zip(*sorted(zip(input_list.flatten(), output_list.flatten())))

    # print('\n')
    # print("input_list_new: ", input_list)
    # print('\n')
    # print("output_list_new: ", output_list)

    input_list = np.array(input_list)
    input_list = np.expand_dims(input_list, axis=0)

    output_list = np.array(output_list)
    output_list = np.expand_dims(output_list, axis=0)

    # print("input_list: ", input_list)
    # print("output_list: ", output_list)

    print("Vocab size: ", vocab_size)
    print("No. of context words: ", output_list.shape[1])

    output_list_one_hot = np.zeros((vocab_size, m), dtype='double')
    output_list_one_hot = output_list_one_hot.T
    # print(output_list_one_hot)
    # print('\n')
    # print(output_list_one_hot.shape)

    for i in range(0, output_list_one_hot.shape[0]):
        # print("i: ", i)
        keyword_num = output_list.flatten()[int(i)]
        # print("keyword_num: ", keyword_num)
        keyword = id_to_word[keyword_num]
        # print("keyword: ", keyword)
        output_list_one_hot[ i ][ word_to_id[keyword] ] = 1

    # print(output_list_one_hot)
    # print('\n')

    input_list_one_hot = np.zeros((vocab_size, m), dtype='double')
    input_list_one_hot = input_list_one_hot.T

    print(input_list_one_hot.shape)

    for i in range(0, output_list_one_hot.shape[0]):
        # print("i: ", i)
        keyword_num = input_list.flatten()[int(i)]
        # print("keyword_num: ", keyword_num)
        keyword = id_to_word[keyword_num]
        # print("keyword: ", keyword)
        input_list_one_hot[ i ][ word_to_id[keyword] ] = 1

    # print(input_list_one_hot)

    # input_list_one_hot_sq = np.zeros((vocab_size, vocab_size))

    input_list_one_hot_sq, ind = np.unique(input_list_one_hot, axis=0, return_index=True)
    input_list_one_hot_sq = input_list_one_hot_sq[np.argsort(ind)]

    return output_list_one_hot, input_list_one_hot, input_list_one_hot_sq, input_list, output_list

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% - Initializing parameters

def initialize_input_layer_matrix(vocab_size, N):

    input_layer = np.random.randn(vocab_size, N) * 0.001
    return input_layer

def initialize_output_layer_matrix(vocab_size, N):
    
    output_layer = np.random.randn(N, vocab_size) * 0.001
    return output_layer

def initialize_parameters(vocab_size, N):

    input_layer_matrix = initialize_input_layer_matrix(vocab_size, N)
    # print(input_layer_matrix)

    output_layer_matrix = initialize_output_layer_matrix(vocab_size, N)
    # print(output_layer_matrix)

    parameters = {}
    parameters['input_layer_matrix'] = input_layer_matrix
    parameters['output_layer_matrix'] = output_layer_matrix

    return parameters

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% - Forward propagation

def forward_input_layer(parameters, row):

    input_layer_matrix = parameters['input_layer_matrix']

    hidden_activation = np.matmul(row, input_layer_matrix)

    return hidden_activation

def forward_output_layer(hidden_activation, parameters):

    output_layer_matrix = parameters['output_layer_matrix']
    output_result = np.matmul(hidden_activation, output_layer_matrix)

    return output_result

def softmax(z):

    softmax_out = np.zeros(z.shape[1])

    n = np.sum(np.exp(z))

    # print('n: ', n)

    if np.isnan(n) or n == 0:
        print('n simple: ', n)
        print('softmax_out: ', softmax_out)
        return [], False

    j = 0
    for i in z.flatten():
        softmax_out[j] = np.divide(np.exp(i), n)

        if np.isnan(softmax_out).any() or n == 0:
            print('n in array: ', n)
            print('softmax_out: ', softmax_out)
            return [], False

        j += 1

    return softmax_out, True

def forward_propagation(parameters, row):

    index = 0
    target = parameters['input_list_one_hot']

    hidden_activation = forward_input_layer(parameters, row[np.newaxis, :])

    output_result = forward_output_layer(hidden_activation, parameters)

    softmax_out, softmax_val = softmax(output_result)

    if softmax_val == False:
        return [], False

    forward_data = {}
    forward_data['hidden_activation_result'] = hidden_activation
    forward_data['output_result'] = output_result
    forward_data['softmax_out'] = softmax_out

    return forward_data, True

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% - Backward propagation

def create_errors(parameters, forward_data, row):

    input_list_one_hot = parameters['input_list_one_hot']
    output_list_one_hot = parameters['output_list_one_hot']
    softmax_out = forward_data['softmax_out']

    # print('input_list_one_hot: ', input_list_one_hot)

    # print('row_back: ', row)

    indexes = np.flatnonzero((input_list_one_hot == row).all(1))

    # print('row index: ', indexes)

    sum_err = np.zeros(softmax_out.shape)

    for i in indexes:
        
        sum_err += softmax_out - output_list_one_hot[i]

    return sum_err, softmax_out

def backward_propagation(parameters, forward_data, row):

    hidden_activation = forward_data['hidden_activation_result']

    input_layer_matrix = parameters['input_layer_matrix']
    output_layer_matrix = parameters['output_layer_matrix']  

    sum_err, softmax_out = create_errors(parameters, forward_data, row)

    w_out_sigma = np.matmul(output_layer_matrix, sum_err)

    grad_w_input = np.matmul(row[np.newaxis, :].T, w_out_sigma[np.newaxis, :])

    grad_w_out = np.matmul(hidden_activation.T, sum_err[np.newaxis, :])

    # print('grad_w_out: ', grad_w_out)
    # print('grad_w_input.shape: ', grad_w_out.shape)

    parameters['input_layer_matrix'] -= learning_rate * grad_w_input
    parameters['output_layer_matrix'] -= learning_rate * grad_w_out

    return parameters, sum_err, softmax_out

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% - Update parameters

def update_parameters(parameters, forward_data, backward_data, learning_rate):

    input_layer_matrix = parameters['input_layer_matrix']
    output_layer_matrix = parameters['output_layer_matrix']

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% - Print functions

def print_params(parameters):

    print("parameters['input_layer_matrix']: ", parameters['input_layer_matrix'])
    print("parameters['input_layer_matrix'].shape: ", parameters['input_layer_matrix'].shape)

    print("parameters['output_layer_matrix']: ", parameters['output_layer_matrix'])
    print("parameters['output_layer_matrix'].shape: ", parameters['output_layer_matrix'].shape)

def print_forward_data(forward_data):

    print("forward_data['hidden_activation_result']: ", forward_data['hidden_activation_result'])
    print("forward_data['hidden_activation_result'].shape: ", forward_data['hidden_activation_result'].shape)

    print("forward_data['output_result']: ", forward_data['output_result'])
    print("forward_data['output_result'].shape: ", forward_data['output_result'].shape)

    print("forward_data['softmax_out']: ", forward_data['softmax_out'])
    print("forward_data['softmax_out'].shape: ", forward_data['softmax_out'].shape)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% - Main

if __name__ == "__main__":

    # Global parameters
    window_size = 3
    N = 100
    whole_time = 0
    ITERATIONS = 2
    count_failed_vectors = 0
    MINIMUM_WORDS = 6

    no_of_documents = 2

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

    print('No. of comments: ', len(df))
    outer_i = 0
    for rows in df.itertuples():
        print('Iteration no.', outer_i, 'out of', len(df))
        word_dictionary = dict()
        words_already_vectorized_file = []
        words_already_vectorized = []
        old_word_vectors = []
        tokens_final = []
        cleaned_text = []
        word_list = []

        if outer_i != 0:
            with open('./IMDB_embeddings/words.txt', 'r') as fin:
                words_already_vectorized_file = fin.readlines()

            if words_already_vectorized_file:
                for line in words_already_vectorized_file:
                    words_already_vectorized.append(line.split(',')[0][2:][:-1])

                print('\n')
                print('words_already_vectorized: ', words_already_vectorized)

            old_word_vectors = np.loadtxt('./IMDB_embeddings/vectors.txt')
            # print('old_word_vectors: ', old_word_vectors)
            # print('old_word_vectors.shape: ', old_word_vectors.shape)

        # print("Word list: ", word_list)
        # print('\n')

        cleaned_text = clear_text(nltk.word_tokenize(rows.Phrase), tokens_final)
        print("Cleaned text: ", cleaned_text)
        print('\n')
        print("Cleaned text length: ", len(cleaned_text))
        print('\n')

        # Valid iteration
        if len(cleaned_text) >= MINIMUM_WORDS:

            # Generating training data
            word_to_id, id_to_word = mapping(cleaned_text)
            # print("Words to id: ", word_to_id)
            # print('\n')
            print("Id to word: ", id_to_word)
            print('\n')
            
            vocab_size = len(id_to_word)

            output_list_one_hot, input_list_one_hot, input_list_one_hot_sq, input_list, output_list = generate_training_data(cleaned_text, word_to_id, id_to_word, window_size, vocab_size)

            parameters = initialize_parameters(vocab_size, N)
            parameters['input_list_one_hot'] = input_list_one_hot
            parameters['input_list_one_hot_sq'] = input_list_one_hot_sq
            parameters['output_list_one_hot'] = output_list_one_hot
            parameters['input_list'] = input_list
            parameters['output_list'] = output_list

            print(parameters)

            old_input_layer_matrix = parameters['input_layer_matrix']

            input_list_one_hot_sq, ind = np.unique(input_list_one_hot, axis=0, return_index=True)
            input_list_one_hot_sq = input_list_one_hot_sq[np.argsort(ind)]

            j = 0
            norm_vec = []

            for row in input_list_one_hot_sq:

                print("Word no. ", j)

                norm_vec = []
                
                learning_rate = 0.05
                t1_inner_start = process_time()
                for i in range(ITERATIONS):

                    # Forward propagation
                    forward_data, forward_val = forward_propagation(parameters, row)

                    if forward_val == False:
                        print('Could not create vector for word no.: ', j)
                        count_failed_vectors += 1
                        break

                    # Backward propagation and parameter update
                    parameters, sum_err, softmax_out = backward_propagation(parameters, forward_data, row)

                    if i % (ITERATIONS // 100) == 0:
                        learning_rate *= 0.98

                    if i % 100 == 0:
                        print('Err. norm: ', np.linalg.norm(sum_err, ord=np.inf))

                        norm_vec.append(np.linalg.norm(sum_err, ord=np.inf))

                # Plotting errors        
                # plt.plot(norm_vec, 'bo')
                # plt.xlabel('Iterációk (x100)')
                # plt.ylabel('Hibavektor normája')
                # plt.axis([0, 10, 0, 0.1])
                # plt.grid(True)
                # plt.show()
                # print('norm_vec: ', norm_vec)

                j += 1
                t1_inner_stop = process_time()
                print("Elapsed time in seconds:", t1_inner_stop - t1_inner_start)
                whole_time += t1_inner_stop - t1_inner_start
            
            # print('Updated params: ')
            # print_params(parameters)

            # If not first valid iteration opnly append and average
            if outer_i != 0:
                # print('old_word_vectors: ', old_word_vectors)

                # print('new vector matrix: ', parameters['input_layer_matrix'])
                index_of_word = 0
                new_index_of_word = len(words_already_vectorized)
                # print('new_index_of_word: ', new_index_of_word)
                print('\n')
                print('Words that already appeared: ')

                for new_word in id_to_word.items():
                    # print(new_word[1])

                    if new_word[1] in words_already_vectorized:
                        print('(', new_word[1], ',', new_word[0], ')')
                        # print(words_already_vectorized.index(new_word[1]))

                        # print(parameters['input_layer_matrix'][new_word[0]])
                        # print(old_word_vectors[words_already_vectorized.index(new_word[1])])

                        old_word_vectors[words_already_vectorized.index(new_word[1])] = (parameters['input_layer_matrix'][new_word[0]] + old_word_vectors[words_already_vectorized.index(new_word[1])]) / 2
                    else:
                        # print('New line to append: ', parameters['input_layer_matrix'][new_word[0]])
                        old_word_vectors = np.vstack([old_word_vectors, parameters['input_layer_matrix'][new_word[0]]])

                        # print('Words that did not appear: ', new_word[1])
                        with open('./IMDB_embeddings/words.txt', 'a') as wout:
                            wout.write(''.join(str([new_word[1], new_index_of_word]) + '\n'))

                        index_of_word += 1
                        new_index_of_word += 1

                # print('old_word_vectors: ', old_word_vectors)

            # If first valid iteration save to file
            if outer_i == 0:
                # Save matrix to file
                np.savetxt('./IMDB_embeddings/vectors.txt', parameters['input_layer_matrix'], fmt='%.8f')

                with open('./IMDB_embeddings/words.txt', 'w') as fout:
                    for key, val in word_to_id.items():
                        fout.write(''.join(str([key, val]) + '\n'))

            else:
                # Save matrix to file
                np.savetxt('./IMDB_embeddings/vectors.txt', old_word_vectors, fmt='%.8f')

            total_words = sum(1 for line in open('./IMDB_embeddings/words.txt'))
            print('Failed to create vectors for', count_failed_vectors, 'words out of', total_words, '->', count_failed_vectors / total_words * 100, '%')
            print("Elapsed time during the whole program in seconds:", whole_time)

            outer_i += 1
            
            # Freeing up memory
            del words_already_vectorized_file, words_already_vectorized
            del word_dictionary, tokens_final, cleaned_text, word_list
            del parameters
            del output_list_one_hot, input_list_one_hot, input_list_one_hot_sq, input_list, output_list
            del old_word_vectors
            gc.collect()

        # Not valid iteration
        else:
            print('Not enough words in phrase no. ', outer_i)
            outer_i += 1
    
    print('Failed to create vectors for a total of', count_failed_vectors, 'words out of', total_words, '->', count_failed_vectors / total_words * 100, '%')