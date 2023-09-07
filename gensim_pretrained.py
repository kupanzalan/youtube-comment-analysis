import gensim.downloader # as api
# from gensim.models.word2vec import Word2Vec
# import inspect
# Show all available models in gensim-data
print(list(gensim.downloader.info()['models'].keys()))

 # Download the "glove-twitter-25" embeddings
glove_vectors = gensim.downloader.load('glove-twitter-25')

# for vector in glove_vectors:
#     print(vector)

# Use the downloaded vectors as usual:
print(glove_vectors.most_similar('twitter'))


# corpus = api.load('text8')
# print(inspect.getsource(corpus.__class__))
# model = Word2Vec(corpus)
# print(model.wv.most_similar('tree'))