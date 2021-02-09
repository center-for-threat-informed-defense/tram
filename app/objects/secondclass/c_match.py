import uuid

from app.utility.base_world import BaseWorld


class Match(BaseWorld):

    @property
    def display(self):
        return self.clean(dict(id=self.id, model=self.model, search=self.search.display, confidence=self.confidence,
                               accepted=self.accepted, sentence=self.sentence, manual=self.manual))

    def __init__(self, model=None, search=None, confidence=0, accepted=True, sentence=None, manual=False):
        self.id = str(uuid.uuid4())
        self.model = model
        self.search = search
        self.confidence = confidence
        self.accepted = accepted
        self.sentence = sentence
        self.manual = manual
