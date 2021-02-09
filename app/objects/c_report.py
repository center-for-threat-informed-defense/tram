import asyncio
import uuid
import nltk
import requests
import urllib3

from bs4 import BeautifulSoup
from enum import Enum

from app.utility.base_world import BaseWorld
from app.utility.file_parser import parse_file

urllib3.disable_warnings()


class Status(Enum):
    QUEUE = 1
    TODO = 2
    REVIEW = 3
    COMPLETED = 4


class Report(BaseWorld):

    MINIMUM_SENTENCE_LENGTH = 4
    EXPORTS = ['default', 'stix']

    @property
    def unique(self):
        return self.id

    @property
    def stage(self):
        return self.status

    @property
    def display(self):
        return self.clean(dict(id=self.unique, status=self.status.name, name=self.name, url=self.url,
                               file=self.file, file_date=self.file_date, exports=self.EXPORTS, matches=[i.display for i in self.matches],
                               sentences=[s.display for s in self.sentences], assigned_user=self.assigned_user))

    def __init__(self, id=None, name=None, url=None, file=None, file_date=None, user=None, status=Status.TODO):
        if type(status) == str:
            status = Status[status]
        elif type(status) == int:
            status = Status(status)

        self.id = id if id else str(uuid.uuid4())
        self.status = status
        self.name = name
        self.url = url
        self.file = file
        self.file_date = file_date
        self.sentences = []
        self.matches = []
        self.completed_models = 0
        self.assigned_user = user

    def store(self, ram):
        existing = self.retrieve(ram['reports'], self.unique)
        if not existing:
            ram['reports'].append(self)
            return self.retrieve(ram['reports'], self.unique)
        existing.update('name', self.name)
        existing.update('assigned_user',self.assigned_user)
        existing.update('status', self.status)
        return existing


    def generate_text_blob(self):
        if self.url:
            content = self._get_soup(self.url)
            return str(content.text), self._clean_url_text(content)
        if self.file:
            content = parse_file(self.file)
            return content, self._clean_file_text(content)

    def export(self, type): # what to return for each export type
        if type == 'stix':
            data = self.display
            output_stix = {}
            output_stix['objects'] = []
            output_stix['id'] = 'bundle--'+data['id']
            for sent in data['sentences']:
                if(len(sent['matches']) > 0):
                    for i in sent['matches']:
                        temp = {}
                        key = 'attack-pattern--'+i['id']
                        temp['type'] = 'attack-pattern'
                        temp['id'] = key
                        temp['name'] = i['search']['name']
                        temp['description'] = sent['text']
                        output_stix['objects'].append(temp)
            for ioc in data['matches']:
                temp = {}
                key = 'indicator--'+ioc['id']
                temp['id'] = key
                temp['indicator_type'] = ioc['search']['code']
                temp['name'] = ioc['search']['description']
                output_stix['objects'].append(temp)
                
            output_stix['type'] = 'bundle'
            output_stix['output_type'] = 'json'
            return output_stix
        else:
            output = self.display
            output['output_type'] = 'json'
            return output

    async def complete(self, total_models):
        self.status = Status.QUEUE
        while self.completed_models < total_models:
            await asyncio.sleep(2)
        self.status = Status.TODO

    """ PRIVATE """

    @staticmethod
    def _get_soup(url):
        r = requests.get(url, verify=False)
        soup = BeautifulSoup(r.content, 'html.parser')
        for tag in ['script', 'style', 'button', 'nav', 'head', 'footer', 'a']:
            [s.extract() for s in soup(tag)]
        return soup

    @staticmethod
    def _clean_file_text(text):
        return nltk.sent_tokenize(text)

    def _clean_url_text(self, soup):
        text_list = []
        for text in soup.text.split('\n'):
            text = text.strip()
            for sentence in nltk.tokenize.sent_tokenize(text):
                if len(nltk.tokenize.word_tokenize(sentence)) > self.MINIMUM_SENTENCE_LENGTH:
                    text_list.append(sentence)
        return text_list
