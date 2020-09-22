import asyncio
import logging
from threading import Thread

from app.services.app_svc import AppService
from app.services.data_svc import DataService
from app.services.file_svc import FileService
from app.services.machine_svc import MachineService
from app.services.tram_svc import TramService
from app.utility.base_world import BaseWorld
from app.objects.c_report import Report
from server import setup_logger


class Tram:
    def __init__(self, logLevel=logging.ERROR):
        setup_logger(level=logLevel)
        
        BaseWorld.apply_config('default', BaseWorld.strip_yml('conf/default.yml')[0])
        BaseWorld.apply_config('regex', BaseWorld.strip_yml('conf/regex.yml')[0])
        version = str(BaseWorld.get_config(prop='version', name='default'))

        self.data_svc = DataService()
        file_svc = FileService()
        tram_svc = TramService()
        machine_svc = MachineService()
        app_svc = AppService(application=None)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(app_svc.verify_dependencies())
        loop.run_until_complete(app_svc.load_data(version))
        loop.run_until_complete(app_svc.load_models(version))

    def create_report(self, report_type='default', **kwargs):
        report = Report(**kwargs)
        machine_svc = MachineService()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(machine_svc.learn(report))
        loop.run_until_complete(self.data_svc.store(report))
        return (report.export(report_type))

    def queue_reports(self, files=[], urls=[]):
        def loop_thread(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        file_reports = [Report(file=f) for f in files]
        url_reports = [Report(url=u) for u in urls]

        machine_svc = MachineService()
        loop = asyncio.get_event_loop()
        loop_t = asyncio.new_event_loop()
        t = Thread(target=loop_thread, args=(loop_t,))
        t.daemon = True
        t.start()
        for report in file_reports + url_reports:
            loop.run_until_complete(self.data_svc.store(report))
        for report in file_reports + url_reports:
            asyncio.run_coroutine_threadsafe(machine_svc.learn(report), loop_t)

        return [(r.id,r.url) for r in url_reports] + [(r.id,r.file) for r in file_reports]

    def status_report(self, report_id):
        return self.get_report(report_id)['status']

    def get_reports(self, report_type='default'):
        loop = asyncio.get_event_loop()
        reports = loop.run_until_complete(self.data_svc.locate('reports'))
        return [r.export(report_type) for r in  reports]

    def get_report(self, report_id, report_type='default'):
        loop = asyncio.get_event_loop()
        reports = loop.run_until_complete(self.data_svc.locate('reports', dict(id=report_id)))
        return reports[0].export(report_type)

    def save_state(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.data_svc.save_state())

    def load_state(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.data_svc.restore_state())

