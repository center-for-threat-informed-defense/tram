from django.core.files.base import File
import pytest

from tram import models


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
def sentence(document, report):
    s = models.Sentence(
        text='test-text',
        document=document,
        order=0,
        report=report,
        disposition=None,
    )
    s.save()
    yield s
    s.delete()


@pytest.fixture
def long_sentence(document, report):
    s = models.Sentence(
        text='this sentence is long and should trigger the overflow',
        document=document,
        order=0,
        report=report,
        disposition=None,
    )
    s.save()
    yield s
    s.delete()


@pytest.fixture
def mapping(report, sentence, attack_technique):
    m = models.Mapping(
        report=report,
        sentence=sentence,
        attack_technique=attack_technique,
        confidence=55.67,
    )
    m.save()
    yield m
    m.delete()


@pytest.mark.django_db
class TestAttackTechnique:
    def test___str__renders_correctly(self, attack_technique):
        # Arrange
        expected = '(T1327) Use multiple DNS infrastructures'

        # Assert
        assert str(attack_technique) == expected


@pytest.mark.django_db
class TestDocument:
    def test__str__renders_correctly(self, document):
        # Arrange
        expected = 'tests/data/simple-test.docx'

        # Assert
        assert str(document) == expected


@pytest.mark.django_db
class TestDocumentProcessingJob:
    def test__str__renders_correctly(self, document_processing_job):
        # Arrange
        expected = 'Process tests/data/simple-test.docx'

        # Assert
        assert str(document_processing_job) == expected


@pytest.mark.django_db
class TestReport:
    def test__str__renders_correctly(self, report):
        # Arrange
        expected = 'Test report name'

        # Assert
        assert str(report) == expected


@pytest.mark.django_db
class TestIndicator:
    def test__str__renders_correctly(self, indicator):
        # Arrange
        expected = 'MD5: 54b0c58c7ce9f2a8b551351102ee0938'

        # Assert
        assert str(indicator) == expected


@pytest.mark.django_db
class TestSentence:
    def test__str__renders_correctly(self, sentence):
        # Arrange
        expected = 'test-text'

        # Assert
        assert str(sentence) == expected

    def test__str__renders_long_sentence_correctly(self, long_sentence):
        # Arrange
        expected = 'this sentence is long and should trigger...'

        # Assert
        assert str(long_sentence) == expected


@pytest.mark.django_db
class TestMapping:
    def test__str__renders_correctly(self, mapping):
        # Arrange
        expected = 'Sentence "test-text" to (T1327) Use multiple DNS infrastructures'

        # Assert
        assert str(mapping) == expected
