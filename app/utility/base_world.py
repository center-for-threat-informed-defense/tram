import logging
import yaml


class BaseWorld:
    """
    A collection of base static functions for service & object module usage
    """

    _app_configuration = dict()

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
