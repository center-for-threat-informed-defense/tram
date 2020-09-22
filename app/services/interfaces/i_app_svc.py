import abc


class AppServiceInterface(abc.ABC):

    @property
    @abc.abstractmethod
    def teardown(self):
        """
        Actions to do when the server stops
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def verify_dependencies(self):
        pass