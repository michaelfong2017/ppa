# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from sqlalchemy.orm import sessionmaker
from scrapy.exceptions import DropItem
from budgetq.models import Question, db_connect, create_table


class SaveQuestionsPipeline(object):
    def __init__(self):
        """
        Initializes database connection and sessionmaker
        Creates tables
        """
        engine = db_connect()
        create_table(engine)
        self.Session = sessionmaker(bind=engine)


    def process_item(self, item, spider):
        """Save quotes in the database
        This method is called for every item pipeline component
        """
        session = self.Session()
        question = Question()
        question.key = item['key']
        question.year = item['year']
        question.bureau = item['bureau']
        question.head = item['head']
        question.head_number = item['head_number']
        question.sub_head = item['sub_head']
        try:
            question.sub_head_number = item['sub_head_number']
        except KeyError:
            pass
        question.controlling_officer_title = item['controlling_officer_title']
        question.controlling_officer_name = item['controlling_officer_name']
        question.programme = item['programme']
        question.reply_serial_no = item['reply_serial_no']
        question.member = item['member']
        question.director = item['director']
        question.member_question_no = item['member_question_no']
        question.question = item['question']
        question.answer = item['answer']
        

        try:
            session.add(question)
            session.commit()

        except:
            session.rollback()
            raise

        finally:
            session.close()

        return item
