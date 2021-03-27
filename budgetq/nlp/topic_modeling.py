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

    def run(self):
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

                index = 0
                for row in records:
                    if index % LOG_EVERY == 0:
                        logger.info(f'Processing row {self.OFFSET + index + 1}')

                    logger.info(row)

                    index = index + 1

                cursor.close()

        finally:
            if conn:
                conn.close()

        logger.info(f'Total time elapsed for processing rows {self.OFFSET + 1} to {self.OFFSET + self.NUMBER_OF_RECORDS}: {datetime.datetime.now() - start_time}')


##
TopicModel.reset_credentials()

##
topic_model = TopicModel(100, 10000)
topic_model.run()

##
topic_model = TopicModel(40, 10)
topic_model.run()

##

