import asyncio
import glob
from importlib import import_module

from app.utility.base_service import BaseService
from app.utility.regex_parser import RegexParser


class MachineService(BaseService):

    def __init__(self):
        self.log = self.add_service('machine_svc', self)
        self.models = self._add_models('app/models', tag='model')
        self.regex_parser = RegexParser()

    async def learn(self, report):
        loop = asyncio.get_event_loop()
        blob, tokens = report.generate_text_blob()
        for regex in self.get_config(name='regex'):
            self.log.debug('[%s] Collecting %s indicator' % (report.id, regex['name']))
            loop.create_task(self.regex_parser.find(regex, report, blob))
        for model in self.models:
            self.log.debug('[%s] Running %s model' % (report.id, model.name))
            try:
                loop.create_task(model.learn(report, tokens))
            except Exception as e:
                self.log.error(e)
        loop.create_task(report.complete(len(self.models)))

    async def retrain(self,model):
        loop = asyncio.get_event_loop()
        self.log.debug('Retraining %s model' % (model))
        try:
            loop.create_task(model[0].train())
        except Exception as e:
            self.log.error(e)

    @staticmethod
    def _add_models(directory, tag):
        models = []
        for filepath in glob.iglob('%s/**.py' % directory):
            module = import_module(filepath.replace('/', '.').replace('\\', '.').replace('.py', ''))
            if tag == 'model':
                models.append(module.Model())
            elif tag == 'indicator':
                models.append(module.Indicator())
        return models
