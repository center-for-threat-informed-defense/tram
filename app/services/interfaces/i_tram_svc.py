import abc


class TramServiceInterface(abc.ABC):

    @abc.abstractmethod
    async def get_reports(self):
        pass

    @abc.abstractmethod
    async def create_report(self, report):
        pass

    @abc.abstractmethod
    async def export_report(self, report_id, type):
        pass
