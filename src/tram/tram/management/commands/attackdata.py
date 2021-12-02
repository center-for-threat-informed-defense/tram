import json

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import IntegrityError

from tram.models import AttackTechnique, AttackGroup

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
        models = [AttackTechnique, AttackGroup]
        for model in models:
            deleted = model.objects.all().delete()
            print(f'Deleted {deleted[0]} {model.__name__} objects')

    def create_attack_technique(self, obj):
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

        t.save()

    def create_attack_group(self, obj):
        g = AttackGroup()
        g.name = obj['name']
        g.stix_id = obj['id']
        for external_reference in obj['external_references']:
            if external_reference['source_name'] not in ('mitre-attack', 'mitre-pre-attack', 'mitre-mobile-attack'):
                continue

            g.attack_id = external_reference['external_id']
            g.attack_url = external_reference['url']
            g.matrix = external_reference['source_name']

        # TODO: Aliases are not saved

        assert g.attack_id is not None
        assert g.attack_url is not None
        assert g.matrix is not None

        g.save()

    def load_attack_data(self, filepath):
        created_stats = {}
        skipped_stats = {}
        integrity_error_stats = {}

        with open(filepath, 'r') as f:
            attack_json = json.load(f)

        assert attack_json['spec_version'] == '2.0'
        assert attack_json['type'] == 'bundle'

        for obj in attack_json['objects']:
            obj_type = obj['type']

            if obj.get('revoked', False):  # Skip revoked objects
                skipped_stats[obj_type] = skipped_stats.get(obj_type, 0) + 1
                continue

            if obj_type == 'attack-pattern':
                try:
                    self.create_attack_technique(obj)
                    created_stats[obj_type] = created_stats.get(obj_type, 0) + 1
                except IntegrityError as e:
                    integrity_error_stats[obj_type] = integrity_error_stats.get(obj_type, 0) + 1
            elif obj_type == 'intrusion-set':
                try:
                    self.create_attack_group(obj)
                    created_stats[obj_type] = created_stats.get(obj_type, 0) + 1
                except IntegrityError as e:
                    integrity_error_stats[obj_type] = integrity_error_stats.get(obj_type, 0) + 1
            else:
                skipped_stats[obj_type] = skipped_stats.get(obj_type, 0) + 1

        print('Load stats for {filepath}:')
        for k, v in created_stats.items():
            print(f'\tCreated {v} {k} objects')
        for k, v in skipped_stats.items():
            print(f'\tSkipped {v} {k} objects')
        for k, v in integrity_error_stats.items():
            print(f'\tIntegrity Error for {v} {k} objects')

    def handle(self, *args, **options):
        subcommand = options['subcommand']

        if subcommand == LOAD:
            self.load_attack_data(settings.DATA_DIRECTORY / 'attack/enterprise-attack.json')
            self.load_attack_data(settings.DATA_DIRECTORY / 'attack/mobile-attack.json')
            self.load_attack_data(settings.DATA_DIRECTORY / 'attack/pre-attack.json')
        elif subcommand == CLEAR:
            self.clear_attack_data()
