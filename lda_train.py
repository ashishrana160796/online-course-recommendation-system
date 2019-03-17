# Import Statements used.
# Libraries used: Pandas, gensim, nltk, pprint.

import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(16)
from gensim import corpora, models
from pprint import pprint

print('Make sure, internet connection is present.') 
print('Downloading wordnet corpus.') 


# Lemmatizing function for inputted text.
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Preprocessing steps after tokenization of document text.
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

# 1. loading data and minor cleaning steps.

print('Loading data into memory.')
course_df = pd.read_csv("data/courses.csv")
print('Dropping rows in any NA values.')
course_df = course_df.dropna(how='any')
course_df['Description'] = course_df['Description'].replace({"'ll": " "}, regex=True)
course_df['CourseId'] = course_df['CourseId'].replace({"-": " "}, regex=True)
course_df['Description'] = course_df.CourseId.str.cat(" "+course_df.CourseTitle.str.cat(" "+course_df.Description))
print('Merging all text related information into one column.')
course_df['Description'] = course_df.Description.replace({"[^A-Za-z0-9 ]+": ""}, regex=True)
course_text = course_df[['Description']]
print('Creating Index for given resulting dataframe.')
course_text['index'] = course_df.index


print('Small peek at dataframe created.')
course_text[:10]

# 2. Load stemmer and preprocess the docs with lemmatization and stemming.
print('Loading Snowball Stemmer into memory.')
stemmer = SnowballStemmer('english')
print('Preprocessing documents with preprocess functions which will lematize and do stemming of given tokens in document.')
processed_docs = course_text['Description'].map(preprocess)
print('Small peek at processed documents created.')
processed_docs[:10]


# 3. Two, approaches being used for training the LDA.
#   a. Bag of Words approach for creating a dictionary based dataset.
#   b. TF-IDF based standard approach where vocabulary created from earlier created BOW corpus.

# a. Bag of Words approach for creating a dictionary based dataset.
print('BOW dataset creation started.')
dictionary = gensim.corpora.Dictionary(processed_docs)
print('Small peek at BOW dataset created.')
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

# Filtering out, from most frequent & least frequent words.
print('Filtering out the extremes present in data from least to most frequents lemmas/stems.')
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


# Result printing from BOW corpus created.
print('Printing results from earlier created BOW corpus.')
bow_doc_786 = bow_corpus[786]
for i in range(len(bow_doc_786)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_786[i][0], 
                                                     dictionary[bow_doc_786[i][0]], 
                                                     bow_doc_786[i][1]))

print('BOW dataset creation and demonstration completed.')


# b. TF-IDF based standard approach. But, BOW corpus created earlier is used for loading TF-IDF values.

print('TF-IDF creation started.')
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

print('Result printing from earlier created TF-IDF corpus.')
for doc in corpus_tfidf:
    pprint(doc)
    break

print('TF-IDF creation and demo completed.')


# 4. Training LDA models based on BOW and TF-IDF methods.

print('Training LDA using BOW model.')
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=17, id2word=dictionary, passes=2, workers=2)

print('Running LDA using BOW model and printing co-efficients associated with each term & related topics.')
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


print('Training LDA using BOW model.')
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=17, id2word=dictionary, passes=2, workers=4)

print('Running LDA using BOW model and printing co-efficients associated with each term & related topics.')
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


# 5. A document test: Machine learning course demo query.

print('A document for query from the corpus. \n')

unseen_doc = ' play by play machine learning exposed Play by Play: Machine Learning Exposed Play by Play is a series in which top technologists work through a problem in real time, unrehearsed, and unscripted. In this course, Play by Play: Machine Learning Exposed, James Weaver and Katharine Beaumont will start with the basics, and build up in an approachable way to some of the most interesting techniques machine learning has to offer. Explore Linear Regression, Neural Networks, clustering, and survey various machine learning APIs and platforms. By the end of this course, you get an overview of what you can achieve, as well as an intuition on the maths behind machine learning.'

print(unseen_doc)

bow_vector = dictionary.doc2bow(preprocess(unseen_doc))
print('Results predicted from BOW model. \n')
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))


print('Results predicted from TF-IDF model. \n')
for index, score in sorted(lda_model_tfidf[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
