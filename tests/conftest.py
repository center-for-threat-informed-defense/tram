import tempfile
from pathlib import Path

import pytest
from django.core.files.base import File
from django.test import override_settings

import tram.settings
from tram import models
from tram.management.commands import attackdata, pipeline


@pytest.fixture(scope="session")
def django_db_setup(django_db_setup, django_db_blocker):
    """
    This fixture overrides key Django settings with values suitable for tests.

    It also loads database fixtures one time so that it doesn't need to be done
    separately for each test. This dramatically improves test execution time.

    This monkey-patching approach is hacky but Django's override_settings() API
    doesn't seem to work for DATA_DIRECTORY or DATABASES.
    """
    with tempfile.TemporaryDirectory() as temp_path:
        tram.settings.DATABASES = {
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        }
        tram.settings.SECRET_KEY = "UNITTEST"
        data_directory = Path(temp_path)
        tram.settings.DATA_DIRECTORY = str(data_directory)
        media_root = data_directory / "media"
        media_root.mkdir(parents=True)

        with django_db_blocker.unblock():
            attackdata.Command().handle(subcommand=attackdata.LOAD)
            pipeline.Command().handle(
                subcommand=pipeline.LOAD_TRAINING_DATA,
                file="tests/data/test-training-data.json",
            )

        settings = {"MEDIA_ROOT": str(media_root), "SECRET_KEY": "UNITTEST"}

        with override_settings(**settings):
            yield


@pytest.fixture
def document():
    with open("tests/data/simple-test.docx", "rb") as f:
        d = models.Document(docfile=File(f))
        d.save()
    yield d
    d.delete()


@pytest.fixture
def attack_object(db):
    """
    $ sqlite3 -line db.sqlite3 "SELECT * FROM tram_attackobject WHERE id=72"
             id = 72
           name = Command and Scripting Interpreter
        stix_id = attack-pattern--7385dfaf-6886-4229-9ecd-6fd678040830
      attack_id = T1059
     attack_url = https://attack.mitre.org/techniques/T1059
         matrix = mitre-attack
     created_on = 2022-03-01 20:06:53.717154
     updated_on = 2022-03-01 20:06:53.717172
    attack_type = technique
      stix_type = attack-pattern
    """
    yield models.AttackObject.objects.get(id=72)


@pytest.fixture
def report(document):
    """
    $ sqlite3 -line db.sqlite3 "SELECT * FROM tram_report WHERE id=1"
               id = 1
             name = Bootstrap Training Data
             text = There is no text for this report. These sentences were mapped by human analysts.
         ml_model = humans
       created_on = 2022-03-01 20:07:05.511904
       updated_on = 2022-03-01 20:07:05.511939
    created_by_id = 1
      document_id =
    """
    yield models.Report.objects.get(id=1)


@pytest.fixture
def document_processing_job(document):
    job = models.DocumentProcessingJob(document=document)
    job.save()
    yield job
    job.delete()


@pytest.fixture
def indicator(report):
    ind = models.Indicator(
        report=report, indicator_type="MD5", value="54b0c58c7ce9f2a8b551351102ee0938"
    )
    ind.save()
    yield ind
    ind.delete()


@pytest.fixture
def sentence():
    """
    $ sqlite3 -line db.sqlite3 "SELECT * FROM tram_sentence WHERE id=33"

             id = 33
           text = !CMD  Trojan executes a command prompt command
          order = 1000
     created_on = 2022-03-01 20:07:05.600772
     updated_on = 2022-03-01 20:07:05.600789
      report_id = 1
    disposition = accept
    document_id =
    """
    yield models.Sentence.objects.get(id=33)


@pytest.fixture()
def short_sentence(report):
    s = models.Sentence(
        text="test-text",
        document=report.document,
        order=0,
        report=report,
        disposition=None,
    )
    s.save()
    yield s
    s.delete()


@pytest.fixture
def long_sentence(report):
    s = models.Sentence(
        text="this sentence is long and should trigger the overflow",
        document=report.document,
        order=0,
        report=report,
        disposition=None,
    )
    s.save()
    yield s
    s.delete()


@pytest.fixture
def mapping():
    """
    $ sqlite3 -line db.sqlite3 "SELECT * FROM tram_mapping WHERE id=33"
                  id = 33
          confidence = 100.0
          created_on = 2022-03-01 20:07:05.602368
          updated_on = 2022-03-01 20:07:05.602386
           report_id = 1
         sentence_id = 33
    attack_object_id = 72
    """
    yield models.Mapping.objects.get(id=33)
