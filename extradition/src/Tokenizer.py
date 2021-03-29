##
# import for database
import sys
import MySQLdb  # conda install -c bioconda mysqlclient
import MySQLdb.cursors

# import for filter
import re

# import for tokenize_and_store
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
import jieba  # conda install -c conda-forge jieba3k

# import for logging
import logging
import datetime

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
class Tokenizer:

    NUMBER_OF_RECORDS = 0
    OFFSET = 0
    user = None
    passwd = None

    stopwords_list = []

    def __init__(self, NUMBER_OF_RECORDS, OFFSET):
        self.NUMBER_OF_RECORDS = NUMBER_OF_RECORDS
        self.OFFSET = OFFSET

    def tokenize_and_store(self, item_data_msg):
        # Store data
        item_data_blockquote_count = 0
        item_data_style_count = 0

        '''
        Filter starts
        '''
        msg = item_data_msg

        # Delete <br />
        new_msg = re.sub(r'<br />', r'', msg)

        # Extract imgs
        imgs = re.findall(r'<img\s.*?src=\"(.*?)[\"\s\n]', new_msg)
        item_data_images = '\n'.join(imgs)

        # Delete img
        new_msg = re.sub(r'<img\s(.*?)/>', r'', new_msg)

        # Extract links
        links = re.findall(r'(?<=<a\shref=\")(.*?)[\"\s\n]', new_msg)
        # Delete links
        new_msg = re.sub(r'<a\s(.|\n)*?</a>', r'', new_msg)

        # Extract links 2 (further extract after deleting)
        links2 = re.findall(r'(?:http|https)[a-zA-Z0-9\.\/\?\:@\-_\=#%\&]*', new_msg)
        # Delete links2

        new_msg = re.sub(r'(?:http|https)[a-zA-Z0-9\.\/\?\:@\-_\=#%\&]*', r'', new_msg)
        # Extract links 3 (further extract after deleting)
        links3 = re.findall(r'(?:t.me\/)[a-zA-Z0-9\.\/\?\:@\-_\=#%\&]*', new_msg)
        item_data_links = '\n'.join(links + links2 + links3)
        # Delete links3
        new_msg = re.sub(r'(?:t.me\/)[a-zA-Z0-9\.\/\?\:@\-_\=#%\&]*', r'', new_msg)

        # Delete <blockquote>, </blockquote>, <span ...>, </span>, <strong>, </strong>,
        # <div style=...>, <ins>, </ins>, <em>, </em>, <pre>, </pre>,
        # <code...>, </code>, <del>, </del>
        # Count </span>, </strong>, <div style=...>, </ins>, </em>, </pre>, </code>, </del>
        subn1 = re.subn(r'<blockquote>', r'', new_msg)
        new_msg = subn1[0]
        subn2 = re.subn(r'</blockquote>', r'', new_msg)
        new_msg = subn2[0]
        item_data_blockquote_count = item_data_blockquote_count + subn2[1]

        subn3 = re.subn(r'<span\sstyle=(.*?)>', r'', new_msg)
        new_msg = subn3[0]
        subn4 = re.subn(r'</span>', r'', new_msg)
        new_msg = subn4[0]
        item_data_style_count = item_data_style_count + subn4[1]

        subn5 = re.subn(r'<strong>', r'', new_msg)
        new_msg = subn5[0]
        subn6 = re.subn(r'</strong>', r'', new_msg)
        new_msg = subn6[0]
        item_data_style_count = item_data_style_count + subn6[1]

        subn7 = re.subn(r'<div\sstyle=(.*?)>', r'', new_msg)
        new_msg = subn7[0]
        subn8 = re.subn(r'</div>', r'', new_msg)
        new_msg = subn8[0]
        item_data_style_count = item_data_style_count + subn8[1]

        subn9 = re.subn(r'<ins>', r'', new_msg)
        new_msg = subn9[0]
        subn10 = re.subn(r'</ins>', r'', new_msg)
        new_msg = subn10[0]
        item_data_style_count = item_data_style_count + subn10[1]

        subn11 = re.subn(r'<em>', r'', new_msg)
        new_msg = subn11[0]
        subn12 = re.subn(r'</em>', r'', new_msg)
        new_msg = subn12[0]
        item_data_style_count = item_data_style_count + subn12[1]

        subn13 = re.subn(r'<pre>', r'', new_msg)
        new_msg = subn13[0]
        subn14 = re.subn(r'</pre>', r'', new_msg)
        new_msg = subn14[0]
        item_data_style_count = item_data_style_count + subn14[1]

        subn15 = re.subn(r'<code(.*?)>', r'', new_msg)
        new_msg = subn15[0]
        subn16 = re.subn(r'</code>', r'', new_msg)
        new_msg = subn16[0]
        item_data_style_count = item_data_style_count + subn16[1]

        subn17 = re.subn(r'<del>', r'', new_msg)
        new_msg = subn17[0]
        subn18 = re.subn(r'</del>', r'', new_msg)
        new_msg = subn18[0]
        item_data_style_count = item_data_style_count + subn18[1]

        filtered_msg = new_msg
        # print(filtered_msg)
        # print(item_data_images)
        # print(item_data_links)
        # print(item_data_blockquote_count)
        # print(item_data_style_count)

        '''
        Filter ends
        '''

        '''
        Tokenizer starts
        '''
        tokens = self.seg_depart(filtered_msg, Tokenizer.stopwords_list)
        return tokens, item_data_images, item_data_links, item_data_blockquote_count, item_data_style_count
        '''
        Tokenizer ends
        '''

    def seg_depart(self, sentence, stopwords_list):
        if not isinstance(sentence, str):
            return ''

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

    @staticmethod
    def read_dicts():
        """Add " 1" to the end of each vocabulary.
        Only do once."""
        with open(os.path.join(FILE_DIR, "data/vocabulary/extradition.txt"), "r") as f:
            lines = f.readlines()
        with open(os.path.join(FILE_DIR, "data/vocabulary/extradition.txt"), "w", encoding='UTF-8') as f:
            f.write('\n'.join([line.rstrip("\n") + (" 1" if not line.rstrip("\n").endswith("1") else "") for line in lines]))

        # Read all dictionaries
        '''
        Load corpora (custom dictionary)
        '''
        start_time = datetime.datetime.now()

        for filename in os.listdir(os.path.join(FILE_DIR, "data/vocabulary")):
            if filename.endswith(".txt"):
                logger.info(f'Loading dictionary {filename}')
                jieba.load_userdict(os.path.join(
                    FILE_DIR, "data/vocabulary/" + filename))

        # read stopwords_list.txt
        logger.info(f'Loading stopwords.txt')
        Tokenizer.stopwords_list = [line.strip() for line in open(os.path.join(
            FILE_DIR, "data/stopwords.txt"), 'r', encoding='UTF-8').readlines()]

        logger.info(f'Time elapsed for loading corpora: {datetime.datetime.now() - start_time}')

    @staticmethod
    def reset_credentials():
        Tokenizer.user = None
        Tokenizer.passwd = None

    def tokenize(self):
        '''
        Retrieve data from MySQL
        '''
        if Tokenizer.user is None:
            Tokenizer.user = input("Enter MySQL username: ")
        if Tokenizer.passwd is None:
            Tokenizer.passwd = input("Enter MySQL user password: ")

        conn = MySQLdb.connect(host='database-1.cfrc4kc4zmgx.ap-southeast-1.rds.amazonaws.com', db='lihkg',
                               user=Tokenizer.user, passwd=Tokenizer.passwd, charset='utf8')

        logger.info(f'NUMBER_OF_RECORDS: {self.NUMBER_OF_RECORDS}')
        logger.info(f'OFFSET: {self.OFFSET}')

        start_time = datetime.datetime.now()

        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f'select title, item_data_msg from processed_data WHERE row_id > {self.OFFSET} ORDER BY row_id LIMIT {self.NUMBER_OF_RECORDS}')
                records = cursor.fetchall()

                logger.info(f'Time elapsed for fetching records: {datetime.datetime.now() - start_time}')
                start_time = datetime.datetime.now()
                middle_time = start_time

                index = 0
                for row in records:
                    if index % LOG_EVERY == 0:
                        logger.info(f'Processing row {self.OFFSET + index + 1}')

                    try:
                        '''
                        row[0] is the title.
                        row[1] is the item_data_msg. We are going to filter, extract and tokenize it,
                        and then store the results in database.
                        '''
                        title_tokens, _, _, _, _ = self.tokenize_and_store(
                            row[0])

                        tokens, item_data_images, item_data_links, item_data_blockquote_count, item_data_style_count = self.tokenize_and_store(
                            row[1])

                        cursor.execute(
                            f'INSERT INTO analyzed_data VALUES (\"{self.OFFSET + index + 1}\", \"{title_tokens}\", \"{tokens}\", \"{item_data_images}\", \"{item_data_links}\", {item_data_blockquote_count}, {item_data_style_count})')

                        conn.commit()

                    except (MySQLdb.Error, MySQLdb.Warning) as e:
                        logger.error(f'{e} for row {self.OFFSET + index + 1}')

                    finally:
                        if index % LOG_EVERY == LOG_EVERY - 1:
                            logger.info(f'Time elapsed for processing rows {self.OFFSET + index + 1 - LOG_EVERY + 1} to {self.OFFSET + index + 1}: {datetime.datetime.now() - middle_time}')
                            middle_time = datetime.datetime.now()

                        index = index + 1

                cursor.close()

        finally:
            if conn:
                conn.close()

        logger.info(f'Total time elapsed for processing rows {self.OFFSET + 1} to {self.OFFSET + self.NUMBER_OF_RECORDS}: {datetime.datetime.now() - start_time}')

##
Tokenizer.read_dicts()

##
Tokenizer.reset_credentials()

##
tokenizer = Tokenizer(10, 0)
tokenizer.tokenize()

##
tokenizer = Tokenizer(174, 5826)
tokenizer.tokenize()

##
for i in range(140):
    Tokenizer(100000, 100000*i).tokenize()

##
