import json
import time

from django.core.files import File
from django.core.management.base import BaseCommand

import tram.models as db_models
from tram.ml import base
from tram import serializers


ADD = 'add'
RUN = 'run'
TEST = 'test'
TRAIN = 'train'
LOAD_TRAINING_DATA = 'load-training-data'


class Command(BaseCommand):
    help = 'Machine learning pipeline commands'

    def add_arguments(self, parser):
        sp = parser.add_subparsers(title='subcommands',
                                   dest='subcommand',
                                   required=True)
        sp_run = sp.add_parser(RUN, help='Run the ML Pipeline')
        sp_run.add_argument('--model', default='tram', help='Select the ML model.')
        sp_test = sp.add_parser(TEST, help='Test the ML pipeline')  # noqa: F841
        sp_test.add_argument('--model', default='tram', help='Select the ML model.')
        sp_train = sp.add_parser(TRAIN, help='Train the ML Pipeline')  # noqa: F841
        sp_train.add_argument('--model', default='tram', help='Select the ML model.')
        sp_add = sp.add_parser(ADD, help='Add a document for processing by the ML pipeline')
        sp_add.add_argument('--file', required=True, help='Specify the file to be added')
        sp_load = sp.add_parser(LOAD_TRAINING_DATA, help='Load training data. Must be formatted as a Report Export.')
        sp_load.add_argument('--file', default='data/training/bootstrap-training-data.json',
                             help='Training data file to load. Defaults: data/training/bootstrap-training-data.json')

    def handle(self, *args, **options):
        subcommand = options['subcommand']

        if subcommand == ADD:
            filepath = options['file']
            with open(filepath, 'rb') as f:
                django_file = File(f)
                db_models.DocumentProcessingJob.create_from_file(django_file)
            self.stdout.write('Added file to ML Pipeline: %s' % filepath)
            return

        if subcommand == LOAD_TRAINING_DATA:
            filepath = options['file']
            self.stdout.write('Loading training data from %s' % filepath)
            with open(filepath, 'r') as f:
                filedata = f.read()
                json_data = json.loads(filedata)
                res = serializers.ReportExportSerializer(data=json_data)
                res.is_valid(raise_exception=True)
                res.save()
            return

        model = options['model']
        model_manager = base.ModelManager(model)

        if subcommand == RUN:
            self.stdout.write('Running ML Pipeline with Model: %s' % model)
            return model_manager.run_model()
        elif subcommand == TEST:
            self.stdout.write('Testing ML Model: %s' % model)
            return model_manager.test_model()
        elif subcommand == TRAIN:
            self.stdout.write('Training ML Model: %s' % model)
            start = time.time()
            return_value = model_manager.train_model()
            end = time.time()
            elapsed = end - start
            self.stdout.write('Trained ML model in %s seconds' % elapsed)
            return return_value
