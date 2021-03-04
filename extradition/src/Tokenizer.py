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
        #
        # '''
        # Filter character by character
        #
        # Keep space when char is symbol or space
        # so that words will not be squeezed together
        # '''
        # sentence = list([char.lower() if char.isalpha() or char.isnumeric() or char == ' '
        #                  else ' ' for char in sentence])
        # sentence = "".join(sentence)

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
        # Read all dictionaries
        '''
        Load corpora (custom dictionary)
        '''
        for filename in os.listdir(os.path.join(FILE_DIR, "data/vocabulary")):
            if filename.endswith(".txt"):
                print(filename)
                jieba.load_userdict(os.path.join(
                    FILE_DIR, "data/vocabulary/" + filename))

        # read stopwords_list.txt
        Tokenizer.stopwords_list = [line.strip() for line in open(os.path.join(
            FILE_DIR, "data/stopwords.txt"), 'r', encoding='UTF-8').readlines()]

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

        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    f'select item_data_post_id, item_data_msg from raw_data where cat_id = 5 LIMIT {self.OFFSET}, {self.NUMBER_OF_RECORDS}')
                records = cursor.fetchall()

                index = 0
                for row in records:
                    if index % 100 == 0:
                        print(f'Now process row with id={index + 1}')

                    try:
                        '''
                        row[1] is the item_data_msg. We are going to filter, extract and tokenize it,
                        and then store the results in database.
                        '''
                        tokens, item_data_images, item_data_links, item_data_blockquote_count, item_data_style_count = self.tokenize_and_store(
                            row[1])

                        cursor.execute(
                            f'INSERT INTO processed_data SELECT *, \"{tokens}\", \"{item_data_images}\", \"{item_data_links}\", {item_data_blockquote_count}, {item_data_style_count} FROM raw_data WHERE cat_id = 5 LIMIT {self.OFFSET + index}, 1')

                        conn.commit()

                    except (MySQLdb.Error, MySQLdb.Warning) as e:
                        print(e)

                    finally:
                        index = index + 1

                cursor.close()

        finally:
            if conn:
                conn.close()

##
Tokenizer.read_dicts()

##
tokenizer = Tokenizer(10, 0)
tokenizer.tokenize()

##
tokenizer = Tokenizer(5, 10)
tokenizer.tokenize()
