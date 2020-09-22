import abc


class DataServiceInterface(abc.ABC):

    @staticmethod
    @abc.abstractmethod
    def destroy():
        """
        Clear out all data
        :return:
        """
        pass

    @abc.abstractmethod
    def save_state(self):
        pass

    @abc.abstractmethod
    def restore_state(self):
        pass

    @abc.abstractmethod
    def store(self, c_object):
        """
        Accept any c_object type and store it (create/update) in RAM
        :param c_object:
        :return: a single c_object
        """
        pass

    @abc.abstractmethod
    def locate(self, object_name, match):
        """
        Find all c_objects which match a search. Return all c_objects if no match.
        :param object_name:
        :param match: dict()
        :return: a list of c_object types
        """
        pass

    @abc.abstractmethod
    def remove(self, object_name, match):
        """
        Remove any c_objects which match a search
        :param object_name:
        :param match: dict()
        :return:
        """
        pass
