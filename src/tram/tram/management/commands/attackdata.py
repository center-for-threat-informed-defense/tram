import json

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.utils import IntegrityError

from tram.models import AttackTechnique

LOAD = 'load'
CLEAR = 'clear'


class Command(BaseCommand):
    help = 'Machine learning pipeline commands'

    def add_arguments(self, parser):
        sp = parser.add_subparsers(title='subcommands',
                                   dest='subcommand',
                                   required=True)
        sp_load = sp.add_parser(LOAD, help='Load ATT&CK Data into the Database')  # noqa: F841
        sp_clear = sp.add_parser(CLEAR, help='Clear ATT&CK Data from the Database')  # noqa: F841

    def clear_attack_data(self):
        AttackTechnique.objects.all().delete()

    def load_attack_data(self, filepath):
        num_revoked = 0
        num_saved = 0

        with open(filepath, 'r') as f:
            attack_json = json.load(f)

        assert attack_json['spec_version'] == '2.0'
        assert attack_json['type'] == 'bundle'

        for obj in attack_json['objects']:
            if obj.get('revoked', False):  # Skip revoked objects
                num_revoked += 1
                continue

            if obj['type'] != 'attack-pattern':  # Skip non-attack patterns
                continue



            t = AttackTechnique()
            t.name = obj['name']
            t.stix_id = obj['id']
            for external_reference in obj['external_references']:
                if external_reference['source_name'] not in ('mitre-attack', 'mitre-pre-attack', 'mitre-mobile-attack'):
                    continue

                t.attack_id = external_reference['external_id']
                t.attack_url = external_reference['url']
                t.matrix = external_reference['source_name']

            assert t.attack_id is not None
            assert t.attack_url is not None
            assert t.matrix is not None

            try:
                t.save()
                num_saved += 1
            except IntegrityError as ex:
                if str(ex) == 'UNIQUE constraint failed: tram_attacktechnique.attack_id':
                    # This attack data has already been loaded; stop processing
                    print(f'The file {filepath} has already been loaded')
                    return
                else:
                    raise ex

    def handle(self, *args, **options):
        subcommand = options['subcommand']

        if subcommand == LOAD:
            self.load_attack_data(settings.DATA_DIRECTORY / 'attack/enterprise-attack.json')
            self.load_attack_data(settings.DATA_DIRECTORY / 'attack/mobile-attack.json')
            self.load_attack_data(settings.DATA_DIRECTORY / 'attack/pre-attack.json')
        elif subcommand == CLEAR:
            self.clear_attack_data()
