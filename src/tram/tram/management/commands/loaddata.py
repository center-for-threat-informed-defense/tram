import json

from django.core.management.base import BaseCommand

from tram import serializers
from tram.management.commands.otxdata import Otxdata


LOAD_TRAINING_DATA = 'load-training-data'
LOAD_OTXDATA = 'otxdata'


class Command(BaseCommand):
    help = 'Machine learning pipeline commands'

    def add_arguments(self, parser):
        sp = parser.add_subparsers(title='subcommands',
                                   dest='subcommand',
                                   required=True)
        sp_load = sp.add_parser(LOAD_TRAINING_DATA, help='Load training data. Must be formatted as a Report Export.')
        sp_load.add_argument('--file', default='data/training/bootstrap-training-data.json',
                             help='Training data file to load. Defaults: data/training/bootstrap-training-data.json')
        sp_otx = sp.add_parser(LOAD_OTXDATA, help='Load otx data for full report model.')
        sp_otx.add_argument('--file', default='data/training/otx-training-data.json',
                            help='Otx data file to load. Defaults: data/training/otx-training-data.json')

    def handle(self, *args, **options):
        subcommand = options['subcommand']

        if subcommand == LOAD_TRAINING_DATA:
            filepath = options['file']
            self.stdout.write(f'Loading training data from {filepath}')
            with open(filepath, 'r') as f:
                res = serializers.ReportExportSerializer(data=json.load(f))
                res.is_valid(raise_exception=True)
                res.save()
            return

        if subcommand == LOAD_OTXDATA:
            print(options)
            filepath = options['file']
            otx = Otxdata()
            otx.load_otx_data(filepath=filepath)
