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
        sp_run.add_argument('--model')
        sp_test = sp.add_parser(TEST, help='Test the ML pipeline')  # noqa: F841
        sp_train = sp.add_parser(TRAIN, help='Train the ML Pipeline')  # noqa: F841

    def handle(self, *args, **options):
        model_manager = ModelManager(model='dummy')

        subcommand = options['subcommand']

        if subcommand == RUN:
            return model_manager.run_model()
        elif subcommand == TEST:
            return self.test_model()
        elif subcommand == TRAIN:
            return self.train_model()

        return
