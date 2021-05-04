#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 23:01:35 2020

@author: michael
"""

def main():
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
    import gensim # conda install -c conda-forge gensim
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

            conn = MySQLdb.connect(host='127.0.0.1', db='budgetq',
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

            logger.info(
                f'Time elapsed for formatting records {self.OFFSET + 1} to {self.OFFSET + self.NUMBER_OF_RECORDS}: {datetime.datetime.now() - start_time}')

            return data_words_list


    ##
    TopicModel.reset_credentials()

    ##
    topic_model = TopicModel(100000, 0)
    data_words_list = topic_model.get_data_words_list()
    logger.info(np.shape(data_words_list))


    ##
    def train(data_words, use_bigram, use_trigram, filter_no_above, num_topics, alpha, eta, random_state, passes, save_filename, topn=20):
        # Group each question and each answer together to improve the score
        # data_words = list(map(lambda t: t[0] + t[1], zip(data_words_list[0], data_words_list[1])))
        # data_words = data_words_list[0] + data_words_list[1]
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        # See trigram example
        # print(bigram_mod[data_words[0]])

        #
        if use_bigram:
            data_lemmatized = [bigram_mod[doc] for doc in data_words]
        elif use_trigram:
            data_lemmatized = [trigram_mod[doc] for doc in data_words]
        else:
            data_lemmatized = data_words

        logger.info(data_lemmatized[:1])

        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)
        '''
        Use no_above to filter stopwords, which are very frequent
        '''
        id2word.filter_extremes(no_above=filter_no_above)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        # View
        logger.info(corpus[:1])

        #
        logger.info([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:2]])

        #
        # Build LDA model
        start_time = datetime.datetime.now()

        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=num_topics,
                                                    random_state=random_state,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=passes,
                                                    # alpha=np.full(num_topics, 0.01),
                                                    # eta=np.full(num_topics, 0.01),
                                                    alpha=alpha,
                                                    eta=eta,
                                                    per_word_topics=True)

        time_elapsed = datetime.datetime.now() - start_time
        logger.info(f'Time elapsed for training LDA model: {time_elapsed}')
        #
        # Print the Keyword in the 20 topics
        for i in range(20):
            pprint(lda_model.print_topic(i))
        doc_lda = lda_model[corpus]

        #
        # Compute Perplexity
        print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)

        #
        import openpyxl

        wb = openpyxl.Workbook()
        ws_write = wb.active
        for i in range(num_topics):
            all_words = lda_model.print_topic(i, topn=topn).split(' + ')
            for j in range(topn):
                cell = ws_write.cell(row=i + 1, column=j + 1)
                cell.value = all_words[j]

        ws_write.cell(row=num_topics+2, column=1).value = 'Perplexity: ' + str(lda_model.log_perplexity(corpus))
        ws_write.cell(row=num_topics+3, column=1).value = 'Coherence Score: ' + str(coherence_lda)

        ws_write.cell(row=num_topics+5, column=1).value = 'Time elapsed for training LDA model: ' + str(time_elapsed)

        ws_write.cell(row=num_topics+7, column=1).value = 'use_bigram: ' + str(use_bigram)
        ws_write.cell(row=num_topics+8, column=1).value = 'use_trigram: ' + str(use_trigram)
        ws_write.cell(row=num_topics+9, column=1).value = 'filter_no_above: ' + str(filter_no_above)
        ws_write.cell(row=num_topics+10, column=1).value = 'num_topics: ' + str(num_topics)
        ws_write.cell(row=num_topics+11, column=1).value = 'alpha: ' + np.array2string(alpha, formatter={'float_kind':lambda x: "%.4f" % x}) if type(alpha) is np.ndarray else str(alpha)
        ws_write.cell(row=num_topics+12, column=1).value = 'eta: ' + np.array2string(eta, formatter={'float_kind':lambda x: "%.4f" % x}) if type(eta) is np.ndarray else str(eta)
        ws_write.cell(row=num_topics+13, column=1).value = 'random_state: ' + str(random_state)
        ws_write.cell(row=num_topics+14, column=1).value = 'passes: ' + str(passes)
        ws_write.cell(row=num_topics+15, column=1).value = 'save_filename: ' + save_filename

        wb.save(filename=save_filename)

    ##
    '''
    Trial 30+
    '''
    i=0
    # data_words = list(map(lambda t: t[0] + t[1], zip(data_words_list[0], data_words_list[1])))
    for fna in [0.1,0.05,0.02,0.01,0.009,0.008,0.007,0.006,0.005,0.004,0.003,
                0.002,0.001,0.0009,0.0008,0.0007,0.0006,0.0005]:
        data_words = data_words_list[0]
        use_bigram = False
        use_trigram = False
        filter_no_above = fna
        num_topics = 25
        # [50,100,200,300,400,500,600,1000]
        alpha = np.full(num_topics, 50/200)
        eta = 'auto'
        save_filename = 'topics_t'+str(30+i)+'.xlsx'
        train(data_words=data_words, use_bigram=use_bigram, use_trigram=use_trigram, filter_no_above=filter_no_above, num_topics=num_topics, alpha=alpha, eta=eta, random_state=1000, passes=2, save_filename=save_filename)

        i = i + 1

if __name__ == '__main__':
    main()