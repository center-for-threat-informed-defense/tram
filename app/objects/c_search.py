from app.objects.interfaces.i_object import FirstClassObjectInterface
from app.utility.base_object import BaseObject


class Search(FirstClassObjectInterface, BaseObject):

    @property
    def unique(self):
        return '%s' % self.id

    @property
    def display(self):
        return self.clean(dict(id=self.unique, tag=self.tag, description=self.description, name=self.name,
                               code=self.code))

    def __init__(self, tag, name=None, description=None, code=None):
        self.id = '%s-%s-%s' % (name, code, description)
        self.tag = tag
        self.description = description
        self.name = name
        self.code = code

    def store(self, ram):
        existing = self.retrieve(ram['search'], self.unique)
        if not existing:
            ram['search'].append(self)
            return self.retrieve(ram['search'], self.unique)
        return existing
