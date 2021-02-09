import re

from app.objects.secondclass.c_match import Match
from app.objects.secondclass.c_sentence import Sentence
from app.utility.base_world import BaseWorld


class Model(BaseWorld):

    def __init__(self):
        self.name = 'regex'

    async def learn(self, report, tokens):
        search = await self.get_service('data_svc').locate('search', dict(tag='attack'))
        for sentence in tokens:
            sen = Sentence(text=sentence)
            try:
                for s in search:
                    for _ in re.findall(r'\b%s\b' % s.code, sentence):
                        sen.matches.append(Match(model=self.name, search=s, confidence=100))
            except Exception as e:
                print(e)
            report.sentences.append(sen)
        report.completed_models += 1
