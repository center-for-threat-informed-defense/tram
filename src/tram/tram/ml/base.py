import json

from tram.models import DocumentProcessingJob, Report

class ModelManager(object):
    def __init__(self, model):
        self.model = model

    def process_job(self, job):
        # TODO: Get file data and such
        report = model.create_report()
        report.save()

    def run_model(self):
        while True:
            job = DocumentProcessingJob.objects.filter().order_by('created_on')
            for job in jobs:
                self.process_job(job)
                job.delete()
            time.sleep(1)
    
    def train_model(self):
        raise NotImplementedError()

    def test_model(self)
        raise NotImplementedError()

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