# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class QuestionItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    key = scrapy.Field()
    year = scrapy.Field()
    bureau = scrapy.Field()
    head = scrapy.Field()
    head_number = scrapy.Field()
    sub_head = scrapy.Field()
    sub_head_number = scrapy.Field()
    controlling_officer_title = scrapy.Field()
    controlling_officer_name = scrapy.Field()
    programme = scrapy.Field()
    reply_serial_no = scrapy.Field()
    member = scrapy.Field()
    director = scrapy.Field()
    member_question_no = scrapy.Field()
    question = scrapy.Field()
    answer = scrapy.Field()
    pass
