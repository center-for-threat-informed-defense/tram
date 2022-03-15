import pytest


@pytest.mark.django_db
class TestAttackTechnique:
    def test___str__renders_correctly(self, attack_object):
        # Arrange
        expected = "(T1059) Command and Scripting Interpreter"

        # Assert
        assert str(attack_object) == expected


@pytest.mark.django_db
class TestDocument:
    def test__str__renders_correctly(self, document):
        # Arrange
        expected = "tests/data/simple-test.docx"

        # Assert
        assert str(document) == expected


@pytest.mark.django_db
class TestDocumentProcessingJob:
    def test__str__renders_correctly(self, document_processing_job):
        # Arrange
        expected = "Process tests/data/simple-test.docx"

        # Assert
        assert str(document_processing_job) == expected


@pytest.mark.django_db
class TestReport:
    def test__str__renders_correctly(self, report):
        # Arrange
        expected = "Bootstrap Training Data"

        # Assert
        assert str(report) == expected


@pytest.mark.django_db
class TestIndicator:
    def test__str__renders_correctly(self, indicator):
        # Arrange
        expected = "MD5: 54b0c58c7ce9f2a8b551351102ee0938"

        # Assert
        assert str(indicator) == expected


@pytest.mark.django_db
class TestSentence:
    def test__str__renders_correctly(self, short_sentence):
        # Arrange
        expected = "test-text"

        # Assert
        assert str(short_sentence) == expected

    def test__str__renders_long_sentence_correctly(self, long_sentence):
        # Arrange
        expected = "this sentence is long and should trigger..."

        # Assert
        assert str(long_sentence) == expected


@pytest.mark.django_db
class TestMapping:
    def test__str__renders_correctly(self, mapping):
        # Arrange
        expected = 'Sentence "!CMD  Trojan executes a command prompt c..." to (T1059) Command and Scripting Interpreter'

        # Assert
        assert str(mapping) == expected
