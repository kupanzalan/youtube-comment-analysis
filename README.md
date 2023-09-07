# YouTube comment analysis



## Getting started

The Data folder contains all the training data. Since I'm using two datasets for creating word embeddings there is a word2vec algorithm for the IMDB and one for the Twitter dataset. The IMDB_embeddings and the Twitter_embeddings directories contain the vectors.txt and words.txt files which are generated after you run the word2vec algorithms. I also tried creating word embeddings using the gensim library, you can find this in the gensim_lib.py file. There are two different classifiers too., one for classifying IMDB comment and on for Twitter comments, The comments_collector.py helps to scrape data from a given YouTube video. 

## How to run the application

In order to run the application you need to have the vectors.txt and the words.txt files generated so that the classifier can train a model using them. After that you need to save the .joblib classification model and then just type ```python3 run.py '{{YouTube video link}}'``` in the console in order to classify some comments.
