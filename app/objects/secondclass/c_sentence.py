import uuid

from app.utility.base_object import BaseObject


class Sentence(BaseObject):

    @property
    def unique(self):
        return self.id

    @property
    def display(self):
        return self.clean(dict(id=self.unique, text=self.text, matches=[m.display for m in self.matches]))

    def __init__(self, id=None, text=None):
        self.id = id if id else str(uuid.uuid4())
        self.text = text
        self.matches = []
