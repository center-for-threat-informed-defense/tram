import abc


class MachineServiceInterface(abc.ABC):

    @abc.abstractmethod
    def learn(self, report):
        pass
