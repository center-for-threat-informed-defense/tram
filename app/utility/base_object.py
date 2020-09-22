from app.utility.base_world import BaseWorld


class BaseObject(BaseWorld):

    schema = None
    display_schema = None
    load_schema = None

    def match(self, criteria):
        if not criteria:
            return self
        criteria_matches = []
        for k, v in criteria.items():
            if type(v) is tuple:
                for val in v:
                    if self.__getattribute__(k) == val:
                        criteria_matches.append(True)
            else:
                if self.__getattribute__(k) == v:
                    criteria_matches.append(True)
        if len(criteria_matches) == len(criteria) and all(criteria_matches):
            return self

    def update(self, field, value):
        if (value or type(value) == list) and (value != self.__getattribute__(field)):
            self.__setattr__(field, value)

    @staticmethod
    def retrieve(collection, unique):
        return next((i for i in collection if i.unique == unique), None)

    @staticmethod
    def hash(s):
        return s

    @staticmethod
    def clean(d):
        for k, v in d.items():
            if v is None:
                d[k] = ''
        return d

    @property
    def display(self):
        if self.display_schema:
            dumped = self.display_schema.dump(self)
        elif self.schema:
            dumped = self.schema.dump(self)
        else:
            raise NotImplementedError
        return self.clean(dumped)

    @classmethod
    def load(cls, dict_obj):
        if cls.load_schema:
            return cls.load_schema.load(dict_obj)
        elif cls.schema:
            return cls.schema.load(dict_obj)
        else:
            raise NotImplementedError
