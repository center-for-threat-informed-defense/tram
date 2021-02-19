import json
import random
import time

from faker import Faker

from tram.models import DocumentProcessingJob, Indicator, Report, Sentence

class ModelManager(object):
    def __init__(self, model=None):
        if model is None or model == 'tram':
            self.model = TramModel()
        elif model == 'dummy':
            self.model = DummyModel()
        else:
            raise ValueError('Unkown model: %s' % model)

    def run_model(self):
        while True:
            jobs = DocumentProcessingJob.objects.all().order_by('created_on')
            for job in jobs:
                report = self.model.create_report(job.document)
                print('Created report %s' % report.name)
                job.delete()
            time.sleep(1)
    
    def train_model(self):
        raise NotImplementedError()

    def test_model(self):
        raise NotImplementedError()


class DummyModel(object):
    def __init__(self):
        self.faker = Faker()

    def train(self):
        pass

    def create_report(self, document):
        """Since this is a Dummy model, ignore the file and 
        create a dummy report
        """
        report_name = 'Report on %s by Dummy Model on 2021-01-01T01:01:01' % document.docfile.name
        report = Report(name=report_name,
                        document=document,
                        ml_model='DummyModel')
        report.save()
        
        for i in range(3):
            ind = Indicator()
            ind.report = report
            ind.indicator_type = 'MD5'
            ind.value = self.faker.md5()
            ind.save()
        
        for i in range(10):
            s = Sentence()
            s.report = report
            s.text = self.faker.sentence()
            s.save()

        return report

    def save_to_file(self, filename):
        pass

    @classmethod
    def load_from_file(filename):
        return DummyModel()


class TramModel(object):
    def train(self):
        """
        Trains the model based on:
          1. ATT&CK data from disk
          2. User-annotated reports from models.Report
        """
        raise NotImplementedError()

    def create_report(self, document):
        """Create a models.Report object, doesn't save it
        """
        raise NotImplementedError()

    def test(self):
        """Returns the f1 score
        """
        raise NotImplementedError()

    def save_to_file(self, filename):
        """Saves this TramModel instance to a pkl file
        """
        raise NotImplementedError

    @classmethod
    def load_from_file(filename):
        """Loads a TramModel instance from pkl file
        """
        raise NotImplementedError()