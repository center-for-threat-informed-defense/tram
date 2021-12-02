import json

from django.conf import settings
from django.core.management.base import BaseCommand

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

    def create_attack_object(self, obj):
        obj_type = obj_type = obj['type']
        if obj_type == 'attack-pattern':
            model_class = AttackTechnique
        elif obj_type == 'intrusion-set':
            model_class = AttackGroup
        else:
            raise ValueError(f'Unsupported ATT&CK object type: {obj_type}')

        for external_reference in obj['external_references']:
            if external_reference['source_name'] not in ('mitre-attack', 'mitre-pre-attack', 'mitre-mobile-attack'):
                continue

            attack_id = external_reference['external_id']
            attack_url = external_reference['url']
            matrix = external_reference['source_name']

        assert attack_id is not None
        assert attack_url is not None
        assert matrix is not None

        obj, created = model_class.objects.get_or_create(
            name=obj['name'],
            stix_id=obj['id'],
            attack_id=attack_id,
            attack_url=attack_url,
            matrix=matrix
        )

        return obj, created

    def load_attack_data(self, filepath):
        created_stats = {}
        skipped_stats = {}

        with open(filepath, 'r') as f:
            attack_json = json.load(f)

        assert attack_json['spec_version'] == '2.0'
        assert attack_json['type'] == 'bundle'

        for obj in attack_json['objects']:
            obj_type = obj['type']

            if obj.get('revoked', False):  # Skip revoked objects
                skipped_stats[obj_type] = skipped_stats.get(obj_type, 0) + 1
                continue

            try:
                db_obj, created = self.create_attack_object(obj)
                if created:
                    created_stats[obj_type] = created_stats.get(obj_type, 0) + 1
                else:
                    skipped_stats[obj_type] = skipped_stats.get(obj_type, 0) + 1
            except ValueError:  # Value error means unsupported object type
                skipped_stats[obj_type] = skipped_stats.get(obj_type, 0) + 1

        print('Load stats for {filepath}:')
        for k, v in created_stats.items():
            print(f'\tCreated {v} {k} objects')
        for k, v in skipped_stats.items():
            print(f'\tSkipped {v} {k} objects')

    def handle(self, *args, **options):
        subcommand = options['subcommand']

        if subcommand == LOAD:
            # Note - as of ATT&CK v8.2
            #   Techniques are unique among files, but
            #   Groups are not unique among files
            self.load_attack_data(settings.DATA_DIRECTORY / 'attack/enterprise-attack.json')
            self.load_attack_data(settings.DATA_DIRECTORY / 'attack/mobile-attack.json')
            self.load_attack_data(settings.DATA_DIRECTORY / 'attack/pre-attack.json')
        elif subcommand == CLEAR:
            self.clear_attack_data()
