import csv
import numpy as np

if __name__ == "__main__":

    with open('./IMDB_embeddings/Already_done/words.txt') as f:
        word_list = f.readlines()

    matrix = np.loadtxt("./IMDB_embeddings/Already_done/vectors.txt")

    print(matrix.shape)

    # print(matrix[38, :])
    word1 = matrix[119, :]
    # print(matrix[25, :])
    word2 = matrix[2181, :]

    cosT = np.inner(word1, word2)
    print('cosT: ', cosT)

    norma1 = np.linalg.norm(word1, ord=2)
    norma2 = np.linalg.norm(word2, ord=2)

    print('norm1: ', norma1)
    print('norm2: ', norma2)

    print('Norm prod: ', norma1 * norma2)

    cos_sim = abs(cosT / (norma1 * norma2))
    print('cos_sim: ', cos_sim)
