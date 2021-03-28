#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 23:01:35 2020

@author: michael
"""

##
# import for database
import sys
import MySQLdb  # conda install -c bioconda mysqlclient
import MySQLdb.cursors

# import for filter
import re

# import for tokenize_and_store
import os

import jieba  # conda install -c conda-forge jieba3k

# import for logging
import logging
import datetime

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import pandas as pd
import numpy as np

from pprint import pprint

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

##
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not len(logger.handlers) == 0:
    logger.handlers.clear()

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('main.log', mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

LOG_EVERY = 1000  # logger.info (processing and time elapsed) every N records

##


class TopicModel:
    NUMBER_OF_RECORDS = 0
    OFFSET = 0
    user = None
    passwd = None

    def __init__(self, number_of_records, offset):
        self.NUMBER_OF_RECORDS = number_of_records
        self.OFFSET = offset

    @staticmethod
    def reset_credentials():
        TopicModel.user = None
        TopicModel.passwd = None

    def get_data_words_list(self):
        """
        Retrieve data from MySQL
        """
        if TopicModel.user is None:
            TopicModel.user = input("Enter MySQL username: ")
        if TopicModel.passwd is None:
            TopicModel.passwd = input("Enter MySQL user password: ")

        conn = MySQLdb.connect(host='localhost', db='budgetq',
                               user=TopicModel.user, passwd=TopicModel.passwd, charset='utf8')

        logger.info(f'NUMBER_OF_RECORDS: {self.NUMBER_OF_RECORDS}')
        logger.info(f'OFFSET: {self.OFFSET}')

        start_time = datetime.datetime.now()

        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f'select tokenized_question, tokenized_answer from question where id > {self.OFFSET} ORDER BY id LIMIT {self.NUMBER_OF_RECORDS}'
                )
                records = cursor.fetchall()

                logger.info(f'Time elapsed for fetching records: {datetime.datetime.now() - start_time}')
                start_time = datetime.datetime.now()

                cursor.close()

        finally:
            if conn:
                conn.close()

        def format_records(q_a):
            return [col.rstrip(';').split(';') for col in q_a]
        data_words_list = list(map(format_records, zip(*records)))

        logger.info(f'Time elapsed for formatting records {self.OFFSET + 1} to {self.OFFSET + self.NUMBER_OF_RECORDS}: {datetime.datetime.now() - start_time}')

        return data_words_list

##
TopicModel.reset_credentials()

##
topic_model = TopicModel(100000, 0)
data_words_list = topic_model.get_data_words_list()
logger.info(np.shape(data_words_list))

##
data_words = data_words_list[0]
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])

##
def make_ngrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

##
data_lemmatized = make_ngrams(data_words)
logger.info(data_lemmatized[:1])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

'''
Use no_above to filter stopwords, which are very frequent
'''
id2word.filter_extremes(no_above=0.005)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
logger.info(corpus[:1])

##
logger.info([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

##
# Build LDA model
start_time = datetime.datetime.now()

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=25,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha=np.empty(25).fill(0.02),
                                           per_word_topics=True)

logger.info(f'Time elapsed for training LDA model: {datetime.datetime.now() - start_time}')
##
# Print the Keyword in the 25 topics
for i in range(25):
    pprint(lda_model.print_topic(i))
doc_lda = lda_model[corpus]

##
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
