import glob

from django.core.files.base import File
import pytest

from tram.management.commands import attackdata, pipeline
from tram import models


@pytest.fixture(scope="session", autouse=True)
def verify_test_data_directory_is_empty(request):
    files = glob.glob('data/media/tests/data/*')
    if len(files) > 0:
        raise ValueError('data/media/tests/data/ is not empty! Remove contents to run tests.')


@pytest.fixture
def load_attack_data():
    command = attackdata.Command()
    command.handle(subcommand=attackdata.LOAD)


@pytest.fixture
def load_small_training_data():
    options = {
        'file': 'data/training/bootstrap-training-data-small.json',
    }
    command = pipeline.Command()
    command.handle(subcommand=pipeline.LOAD_TRAINING_DATA, **options)


@pytest.fixture
def document():
    with open('tests/data/simple-test.docx', 'rb') as f:
        d = models.Document(docfile=File(f))
        d.save()
    yield d
    d.delete()


@pytest.fixture
def attack_technique():
    at = models.AttackTechnique(
        name='Use multiple DNS infrastructures',
        stix_id='attack-pattern--616238cb-990b-4c71-8f50-d8b10ed8ce6b',
        attack_id='T1327',
        attack_url='https://attack.mitre.org/techniques/T1327',
        matrix='mitre-pre-attack',
    )
    at.save()
    yield at
    at.delete()


@pytest.fixture
def report(document):
    rpt = models.Report(
        name='Test report name',
        document=document,
        text='test-document-text',
    )
    rpt.save()
    yield rpt
    rpt.delete()


@pytest.fixture
def document_processing_job(document):
    job = models.DocumentProcessingJob(document=document)
    job.save()
    yield job
    job.delete()


@pytest.fixture
def indicator(report):
    ind = models.Indicator(
        report=report,
        indicator_type='MD5',
        value='54b0c58c7ce9f2a8b551351102ee0938'
    )
    ind.save()
    yield ind
    ind.delete()


@pytest.fixture
def sentence(report):
    s = models.Sentence(
        text='test-text',
        document=report.document,
        order=0,
        report=report,
        disposition=None,
    )
    s.save()
    yield s
    s.delete()


@pytest.fixture
def simple_training_data(report, load_attack_data):
    s = models.Sentence(
        text='test-text',
        report=report,
        document=report.document,
        disposition='accept',
    )
    s.save()
    at = models.AttackTechnique.objects.get(attack_id='T1327')
    m = models.Mapping(
        report=report,
        sentence=s,
        attack_technique=at,
        confidence=55.55,
    )
    m.save()
    yield
    m.delete()
    s.delete()


@pytest.fixture
def long_sentence(report):
    s = models.Sentence(
        text='this sentence is long and should trigger the overflow',
        document=report.document,
        order=0,
        report=report,
        disposition=None,
    )
    s.save()
    yield s
    s.delete()


@pytest.fixture
def sentence_technique_mapping(report, sentence, attack_technique):
    m = models.SentenceTechniqueMapping(
        report=report,
        sentence=sentence,
        attack_technique=attack_technique,
        confidence=55.67,
    )
    m.save()
    yield m
    m.delete()
