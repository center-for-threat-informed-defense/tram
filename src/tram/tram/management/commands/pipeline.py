from django.core.management.base import BaseCommand

from tram.ml.base import ModelManager

RUN = 'run'
TEST = 'test'
TRAIN = 'train'


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

    def handle(self, *args, **options):
        model = options['model']
        model_manager = ModelManager(model)

        subcommand = options['subcommand']

        if subcommand == RUN:
            print('Running ML Pipeline with Model: %s' % model)
            return model_manager.run_model()
        elif subcommand == TEST:
            print('Testing ML Model: %s' % model)
            return model_manager.test_model()
        elif subcommand == TRAIN:
            print('Training ML Model: %s' % model)
            return model_manager.train_model()

        return
