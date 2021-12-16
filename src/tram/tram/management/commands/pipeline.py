import json
import time

from django.core.files import File
from django.core.management.base import BaseCommand


from tram.ml import base
import tram.models as db_models
from tram import serializers


ADD = 'add'
RUN = 'run'
TRAIN = 'train'
LOAD_TRAINING_DATA = 'load-training-data'
EXPORT = 'export'


class Command(BaseCommand):
    help = 'Machine learning pipeline commands'

    def add_arguments(self, parser):
        sp = parser.add_subparsers(title='subcommands',
                                   dest='subcommand',
                                   required=True)
        sp_run = sp.add_parser(RUN, help='Run the ML Pipeline')
        sp_run.add_argument('--model', default='logreg', help='Select the ML model.')
        sp_run.add_argument('--run-forever', default=False, action='store_true',
                            help='Specify whether to run forever, or quit when there are no more jobs to process')
        sp_train = sp.add_parser(TRAIN, help='Train the ML Pipeline')  # noqa: F841
        sp_train.add_argument('--model', default='logreg', help='Select the ML model.')
        sp_add = sp.add_parser(ADD, help='Add a document for processing by the ML pipeline')
        sp_add.add_argument('--file', required=True, help='Specify the file to be added')
        sp_load = sp.add_parser(LOAD_TRAINING_DATA, help='Load training data. Must be formatted as a Report Export.')
        sp_load.add_argument('--file', default='data/training/bootstrap-training-data.json',
                             help='Training data file to load. Defaults: data/training/bootstrap-training-data.json')
        sp_export = sp.add_parser(EXPORT,help='export model data to json file or screen')
        sp_export.add_argument('--file', action="store_true",
                             help='If flag is specific will output data to data/model-output.json')

    def handle(self, *args, **options):
        subcommand = options['subcommand']

        if subcommand == ADD:
            filepath = options['file']
            with open(filepath, 'rb') as f:
                django_file = File(f)
                db_models.DocumentProcessingJob.create_from_file(django_file)
            self.stdout.write(f'Added file to ML Pipeline: {filepath}')
            return

        if subcommand == LOAD_TRAINING_DATA:
            filepath = options['file']
            self.stdout.write(f'Loading training data from {filepath}')
            with open(filepath, 'r') as f:
                res = serializers.ReportExportSerializer(data=json.load(f))
                res.is_valid(raise_exception=True)
                res.save()
            return

        model = options['model']
        model_manager = base.ModelManager(model)

        if subcommand == RUN:
            self.stdout.write(f'Running ML Pipeline with Model: {model}')
            return model_manager.run_model(options['run_forever'])
        elif subcommand == TRAIN:
            self.stdout.write(f'Training ML Model: {model}')
            start = time.time()
            return_value = model_manager.train_model()
            end = time.time()
            elapsed = end - start
            self.stdout.write(f'Trained ML model in {elapsed} seconds')
            return return_value
        elif subcommand == EXPORT:
            adv_reports = db_models.Report.objects.filter(ml_model='adversary')
            full_reports = db_models.Report.objects.filter(ml_model='fullreport')
            for report_obj in adv_reports:  # Only store sentences with a labeled technique
                adv_mappings = db_models.AdversaryMapping.objects.filter(report=report_obj)
            
            for report_obj in full_reports:
                full_mappings = db_models.Mapping.objects.filter(report=report_obj)

            adv_json = serializers.serialize("json",adv_mappings)
            full_json = serializers.serialize("json",full_mappings)
            output_json = {
                "Adversary": adv_json,
                "Full Report": full_json
            }
            if options['file']:
                with open('data/model-output.json','w') as f:
                    f.write(output_json)
            else:
                print(output_json)

