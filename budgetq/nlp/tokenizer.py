#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 23:01:35 2020

@author: michael
"""

# %%
import MySQLdb
import MySQLdb.cursors
import pandas as pd
import re
import en_core_web_lg
# %%
from nltk.corpus import *
import nltk
import jieba
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import corpora, models, similarities

# %%
import os
FILE_DIR = os.path.dirname(os.path.abspath('__file__'))

# #%% Only called once when " 1" has not been appended in dictionary.
# with open(os.path.join(FILE_DIR, "data/vocabulary/cantonese_dict.txt")) as f:
#     lines = f.read().splitlines()
# with open(os.path.join(FILE_DIR, "data/vocabulary/cantonese_dict.txt"), "w") as f:
#     f.write('\n'.join([line + ' 1' for line in lines]))

# #%% Only called once when stopwords.txt still includes duplicates.
# with open(os.path.join(FILE_DIR, "data/stopwords.txt")) as f:
#     lines = f.read().splitlines()
# s = set(lines)
# with open(os.path.join(FILE_DIR, "data/stopwords.txt"), "w") as f:
#     f.write('\n'.join([line for line in list(s)]))

# %%
nltk.download('all-corpora')

# %%
'''
Download NLTK corpora
'''
nltk_corpora = ["unicode_samples", "indian", "stopwords", "brown", "swadesh",
                "mac_morpho", "abc", "words", "udhr2", "lin_thesaurus", "webtext",
                "names", "sentiwordnet", "cmudict", "ptb", "inaugural", "conll2002",
                "ieer", "problem_reports", "floresta", "sinica_treebank", "gutenberg",
                "kimmo", "nonbreaking_prefixes", "senseval", "verbnet", "chat80",
                "biocreative_ppi", "framenet_v17", "pil", "alpino", "omw", "cess_cat",
                "shakespeare", "city_database", "movie_reviews", "wordnet_ic",
                "conll2000", "dependency_treebank", "wordnet", "cess_esp", "toolbox",
                "mte_teip5", "treebank", "rte", "nps_chat", "crubadan", "ppattach",
                "switchboard", "brown_tei", "verbnet3", "ycoe", "timit", "pl196x",
                "state_union", "framenet_v15", "paradigms", "genesis", "gazetteers",
                "qc", "udhr", "dolch"]

for corpus in nltk_corpora:
    try:
        exec("word_list = %s.words()" % corpus)
        if not corpus == "stopwords":
            with open(os.path.join(FILE_DIR, "data/vocabulary/nltk_"+corpus+".txt"), "w") as f:
                f.write('\n'.join([line + " 1" for line in word_list]))
    except:
        print("ImportError")


# %%
'''
Download Spacy corpora
'''
nlp = en_core_web_lg.load()
with open(os.path.join(FILE_DIR, "data/vocabulary/spacy_en_core_web_lg.txt"), "w") as f:
    f.write('\n'.join([line + " 1" for line in list(nlp.vocab.strings)]))


# %%
'''
Load corpora (custom dictionary)
'''
for filename in os.listdir(os.path.join(FILE_DIR, "data/vocabulary")):
    if filename.endswith(".txt"):
        print(filename)
        jieba.load_userdict(os.path.join(
            FILE_DIR, "data/vocabulary/" + filename))

# %%
stopwords_list = [line.strip() for line in open(os.path.join(
    FILE_DIR, "data/stopwords.txt"), 'r', encoding='UTF-8').readlines()]
filter_list = [line.strip() for line in open(os.path.join(
    FILE_DIR, "data/filter.txt"), 'r', encoding='UTF-8').readlines()]


# %%


def seg_depart(sentence, stopwords_list):
    if not isinstance(value, str):
        return ''

    '''
    Filter patterns such as <br/>, <table border="
    '''
    for f in filter_list:
        sentence = re.sub(f, '', sentence)

    '''
    Filter character by character
    
    Keep space when char is symbol or space 
    so that words will not be squeezed together
    '''
    sentence = list([char.lower() if char.isalpha() or char.isnumeric() or char == ' '
                     else ' ' for char in sentence])
    sentence = "".join(sentence)

    '''Tokenization'''
    sentence_depart = jieba.cut(sentence.strip())

    outstr = ''

    for word in sentence_depart:
        '''
        Remove punctuations and stopwords
        '''
        if (word.isalpha() or word.isdigit()) and word not in stopwords_list:
            outstr += word
            outstr += ";"
    return outstr


# %%

col_list = ["id", "question", "answer"]
df_qa = pd.read_csv(os.path.join(
    FILE_DIR, "data/questions.csv"), sep=";", usecols=col_list)

# %%
# i = 0
# for index, value in df_qa['question'].iteritems():
#     if i >= 10:
#         break
#     # print(value)
#     tokens = seg_depart(value, stopwords_list)
#     print(tokens)
#     i = i + 1

# %%
print(df_qa['question'][1])

# %%
'''Add columns in MySQL first if not done'''
'''
alter table budgetq.question
add column tokenized_question text;
alter table budgetq.question
add column tokenized_answer text;
'''

# %%
conn = MySQLdb.connect(host='localhost', db='budgetq',
                       user='root', passwd='P@ssw0rd', charset='utf8')

try:
    with conn.cursor() as cursor:
        cursor.execute('SET SQL_SAFE_UPDATES=0')

        for index, value in df_qa['question'].iteritems():
            # if index >= 100:
            #     break
            if index % 500 == 0:
                print(f'Now process row with id={index+1}')
            tokens = seg_depart(value, stopwords_list)

            cursor.execute(
                f'UPDATE question Q SET Q.tokenized_question=\"{tokens}\" WHERE Q.id=\"{index+1}\"')
            cursor.execute('SET SQL_SAFE_UPDATES=1')

            conn.commit()

            i = i + 1

finally:
    conn.close()

# %%
conn = MySQLdb.connect(host='localhost', db='budgetq',
                       user='root', passwd='P@ssw0rd', charset='utf8')

try:
    with conn.cursor() as cursor:
        cursor.execute('SET SQL_SAFE_UPDATES=0')

        for index, value in df_qa['answer'].iteritems():
            # if index >= 100:
            #     break
            if index % 500 == 0:
                print(f'Now process row with id={index+1}')
            tokens = seg_depart(value, stopwords_list)

            cursor.execute(
                f'UPDATE question Q SET Q.tokenized_answer=\"{tokens}\" WHERE Q.id=\"{index+1}\"')
            cursor.execute('SET SQL_SAFE_UPDATES=1')

            conn.commit()

            i = i + 1

finally:
    conn.close()
