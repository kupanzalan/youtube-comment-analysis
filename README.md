# YouTube comment analysis

This Python-based project is focused on sentiment analysis for YouTube comments. The main goal of this project was to analyze and classify the sentiment of user comments on YouTube videos. To achieve this, I implemented the Word2Vec algorithm to create word vectors, leveraging pre-labeled datasets from IMDB and Twitter to train the model.

The process involved the following key steps:

Word2Vec Implementation: I trained the Word2Vec model on the IMDB and Twitter datasets to generate high-quality word vectors. This allowed me to capture semantic relationships between words, which is crucial for accurate sentiment analysis.

Classifier Construction: Using the word vectors, I built a classifier to predict the sentiment of unseen data. The classifier was trained on a labeled training set and evaluated on a separate test set.

YouTube Comment Classification: After training the model, I applied it to classify sentiments in YouTube comments. The classification model determines whether the sentiment of a given comment is positive, negative, or neutral.

This project demonstrates my ability to apply advanced natural language processing (NLP) techniques and machine learning algorithms to real-world datasets. It showcases my expertise in text data processing, sentiment analysis, and building classification models.

## Getting started

The Data folder contains all the training data. Since I'm using two datasets for creating word embeddings there is a word2vec algorithm for the IMDB and one for the Twitter dataset. The IMDB_embeddings and the Twitter_embeddings directories contain the vectors.txt and words.txt files which are generated after you run the word2vec algorithms. I also tried creating word embeddings using the gensim library, you can find this in the gensim_lib.py file. There are two different classifiers too., one for classifying IMDB comment and on for Twitter comments, The comments_collector.py helps to scrape data from a given YouTube video. 

## How to run the application

In order to run the application you need to have the vectors.txt and the words.txt files generated so that the classifier can train a model using them. After that you need to save the .joblib classification model and then just type 

```python3 run.py '{{YouTube video link}}'```

in the console in order to classify some comments.
