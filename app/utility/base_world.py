import logging
import yaml


class BaseWorld(object):
    """
    A collection of base static functions for service & object module usage
    """

    _app_configuration = dict()
    _services = dict()  # From old BaseService class
    # Properties from old BaseObject class
    schema = None
    display_schema = None
    load_schema = None

    @staticmethod
    def apply_config(name, config):
        BaseWorld._app_configuration[name] = config

    @staticmethod
    def get_config(prop=None, name=None):
        name = name if name else 'default'
        if prop:
            return BaseWorld._app_configuration[name].get(prop)
        return BaseWorld._app_configuration[name]

    @staticmethod
    def set_config(name, prop, value):
        if value is not None:
            logging.debug('Configuration (%s) update, setting %s=%s' % (name, prop, value))
            BaseWorld._app_configuration[name][prop] = value

    @staticmethod
    def create_logger(name):
        return logging.getLogger(name)

    @staticmethod
    def strip_yml(path):
        if path:
            with open(path, encoding='utf-8') as seed:
                return list(yaml.load_all(seed, Loader=yaml.FullLoader))
        return []

    # Functions from old BaseService class
    def add_service(self, name, svc):
        self.__class__._services[name] = svc
        return self.create_logger(name)

    @classmethod
    def get_service(cls, name):
        return cls._services.get(name)

    @classmethod
    def get_services(cls):
        return cls._services
    # End functions from old BaseService class

    # Functions from old BaseObject class
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
    # Functions from old BaseObject class
