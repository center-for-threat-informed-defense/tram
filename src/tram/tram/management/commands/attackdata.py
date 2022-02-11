import json
import logging

from django.conf import settings
from django.core.management.base import BaseCommand

from tram.models import AttackObject

LOAD = "load"
CLEAR = "clear"
logger = logging.getLogger(__name__)


STIX_TYPE_TO_ATTACK_TYPE = {
    "attack-pattern": "technique",
    "course-of-action": "mitigation",
    "intrusion-set": "group",
    "malware": "software",
    "tool": "software",
    "x-mitre-tactic": "tactic",
}


class Command(BaseCommand):
    help = "Machine learning pipeline commands"

    def add_arguments(self, parser):
        sp = parser.add_subparsers(
            title="subcommands", dest="subcommand", required=True
        )
        sp_load = sp.add_parser(  # noqa: F841
            LOAD, help="Load ATT&CK Data into the Database"
        )
        sp_clear = sp.add_parser(  # noqa: F841
            CLEAR, help="Clear ATT&CK Data from the Database"
        )

    def clear_attack_data(self):
        deleted = AttackObject.objects.all().delete()
        logger.info("Deleted %d Attack objects", deleted[0])

    def create_attack_object(self, obj):
        for external_reference in obj["external_references"]:
            if external_reference["source_name"] not in (
                "mitre-attack",
                "mitre-pre-attack",
                "mitre-mobile-attack",
            ):
                continue

            attack_id = external_reference["external_id"]
            attack_url = external_reference["url"]
            matrix = external_reference["source_name"]

        assert attack_id is not None
        assert attack_url is not None
        assert matrix is not None

        stix_type = obj["type"]
        attack_type = STIX_TYPE_TO_ATTACK_TYPE[stix_type]

        obj, created = AttackObject.objects.get_or_create(
            name=obj["name"],
            stix_id=obj["id"],
            stix_type=stix_type,
            attack_id=attack_id,
            attack_type=attack_type,
            attack_url=attack_url,
            matrix=matrix,
        )

        return obj, created

    def load_attack_data(self, filepath):
        created_stats = {}
        skipped_stats = {}

        with open(filepath, "r") as f:
            attack_json = json.load(f)

        assert attack_json["spec_version"] == "2.0"
        assert attack_json["type"] == "bundle"

        for obj in attack_json["objects"]:
            obj_type = obj["type"]

            # TODO: Skip deprecated objects
            if obj.get("revoked", False):  # Skip revoked objects
                skipped_stats[obj_type] = skipped_stats.get(obj_type, 0) + 1
                continue

            if obj_type in (
                "relationship",
                "course-of-action",
                "identity",
                "x-mitre-matrix",
                "marking-definition",
            ):
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

        logger.info("Load stats for %s:", filepath)
        for k, v in created_stats.items():
            logger.info("Created %s %s objects", v, k)
        for k, v in skipped_stats.items():
            logger.info("Skipped %s %s objects", v, k)

    def handle(self, *args, **options):
        subcommand = options["subcommand"]

        if subcommand == LOAD:
            # Note - as of ATT&CK v8.2
            #   Techniques are unique among files, but
            #   Groups are not unique among files
            self.load_attack_data(
                settings.DATA_DIRECTORY / "attack/enterprise-attack.json"
            )
            self.load_attack_data(settings.DATA_DIRECTORY / "attack/mobile-attack.json")
            self.load_attack_data(settings.DATA_DIRECTORY / "attack/pre-attack.json")
        elif subcommand == CLEAR:
            self.clear_attack_data()
