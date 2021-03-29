import warnings

from django.core.files.base import File
import pytest

from tram.ml import base
import tram.models as db_models


@pytest.fixture
def dummy_model():
    return base.DummyModel()


@pytest.fixture
def tram_model():
    return base.TramModel()


class TestIndicator:
    def test_repr_is_correct(self):
        # Arrange
        expected = 'Indicator: MD5=54b0c58c7ce9f2a8b551351102ee0938'

        # Act
        ind = base.Indicator(type_='MD5',
                             value='54b0c58c7ce9f2a8b551351102ee0938')

        # Assert
        assert str(ind) == expected


class TestSentence:
    def test_sentence_stores_no_mapping(self):
        # Arrange
        text = 'this is text'
        order = 0
        mappings = None

        # Arraange / Act
        s = base.Sentence(text, order, mappings)

        # Assert
        assert s.text == text
        assert s.order == order
        assert s.mappings == mappings


class TestMapping:
    def test_mapping_repr_is_correct(self):
        # Arrange
        confidence = 95.342000
        attack_technique = 'T1327'
        expected = 'Confidence=95.342000; Technique=T1327'

        # Act
        m = base.Mapping(confidence, attack_technique)

        assert str(m) == expected


class TestReport:
    def test_report_stores_properties(self):
        # Arrange
        name = 'Test report'
        text = 'Test report text'
        sentences = [
            base.Sentence('test sentence text', 0, None)
        ]
        indicators = [
            base.Indicator('MD5', '54b0c58c7ce9f2a8b551351102ee0938')
        ]

        # Act
        rpt = base.Report(
            name=name,
            text=text,
            sentences=sentences,
            indicators=indicators
        )

        # Assert
        assert rpt.name == name
        assert rpt.text == text
        assert rpt.sentences == sentences
        assert rpt.indicators == indicators


@pytest.mark.django_db
class TestModel:
    """Tests ml.base.Model via DummyModel"""

    def test__sentence_tokenize_works_for_paragraph(self, dummy_model):
        # Arrange
        paragraph = """Hello. My name is test. I write sentences. Tokenize, tokenize, tokenize!
                    When will this entralling text stop, praytell? Nobody knows; the author can't stop.
                    """
        expected = ['Hello.', 'My name is test.', 'I write sentences.',
                    'Tokenize, tokenize, tokenize!', 'When will this entralling text stop, praytell?',
                    'Nobody knows; the author can\'t stop.']

        # Act
        sentences = dummy_model._sentence_tokenize(paragraph)

        # Assert
        assert expected == sentences

    @pytest.mark.parametrize("filepath,expected", [
        ('tests/data/AA20-302A.pdf', 'GLEMALT With a Ransomware Chaser'),
        ('tests/data/AA20-302A.docx', 'Page 22 of 22 | Product ID: AA20-302A  TLP:WHITE'),
        ('tests/data/AA20-302A.html', 'CISA is part of the Department of Homeland Security'),
    ])
    def test__extract_text_succeeds(self, dummy_model, filepath, expected):
        # Arrange
        with open(filepath, 'rb') as f:
            doc = db_models.Document(docfile=File(f))
            doc.save()

        # Act
        text = dummy_model._extract_text(doc)

        # Cleanup
        doc.delete()

        # Assert
        assert expected in text

    def test__extract_text_unknown_extension_raises_value_error(self, dummy_model):
        # Arrange
        with open('tests/data/unknown-extension.fizzbuzz', 'rb') as f:
            doc = db_models.Document(docfile=File(f))
            doc.save()

        # Act / Assert
        with pytest.raises(ValueError):
            dummy_model._extract_text(doc)

        # Cleanup
        doc.delete()

    def test_get_report_name_succeeds(self, dummy_model):
        # Arrange
        expected = 'Report for AA20-302A'
        with open('tests/data/AA20-302A.docx', 'rb') as f:
            doc = db_models.Document(docfile=File(f))
            doc.save()
        job = db_models.DocumentProcessingJob(document=doc)
        job.save()

        # Act
        report_name = dummy_model._get_report_name(job)

        # Cleanup
        job.delete()
        doc.delete()

        # Assert
        assert report_name.startswith(expected)

    def test_get_attack_techniques_raises_if_not_initialized(self, dummy_model):
        # Act / Assert
        with pytest.raises(ValueError):
            dummy_model.get_attack_technique_ids()

    @pytest.mark.usefixtures('load_attack_data')
    def test_get_attack_techniques_succeeds_after_initialization(self, dummy_model):
        # Act
        techniques = dummy_model.get_attack_technique_ids()

        # Assert
        assert 'T1327' in techniques  # Ensures mitre-pre-attack is available
        assert 'T1497.003' in techniques  # Ensures mitre-attack is available
        assert 'T1579' in techniques  # Ensures mitre-mobile-attack is available

    @pytest.mark.usefixtures('load_attack_data')
    def test_disk_round_trip_succeeds(self, dummy_model, tmpdir):
        # Arrange
        filepath = (tmpdir + 'dummy_model.pkl').strpath

        # Act
        dummy_model.get_attack_technique_ids()  # Change the state of the DummyModel
        dummy_model.save_to_file(filepath)

        dummy_model_2 = base.DummyModel.load_from_file(filepath)

        # Assert
        assert dummy_model.__class__ == dummy_model_2.__class__
        assert dummy_model.get_attack_technique_ids() == dummy_model_2.get_attack_technique_ids()

    @pytest.mark.usefixtures('load_attack_data')
    def test_process_job_produces_valid_report(self, dummy_model):
        # Arrange
        with open('tests/data/AA20-302A.docx', 'rb') as f:
            doc = db_models.Document(docfile=File(f))
            doc.save()
        job = db_models.DocumentProcessingJob(document=doc)
        job.save()
        # Act
        report = dummy_model.process_job(job)

        # Cleanup
        job.delete()
        doc.delete()

        # Assert
        assert report.name is not None
        assert report.text is not None
        assert len(report.sentences) > 0
        assert len(report.indicators) > 0

    def test_no_data_get_training_data_succeeds(self, dummy_model):
        # Act
        training_data = dummy_model.get_training_data()

        # Assert
        assert len(training_data) == 0

    def test_get_training_data_returns_only_accepted_sentences(self, dummy_model, report):
        # Arrange
        s1 = db_models.Sentence.objects.create(
            text='sentence1',
            order=0,
            document=report.document,
            report=report,
            disposition=None
        )
        s2 = db_models.Sentence.objects.create(
            text='sentence 2',
            order=1,
            document=report.document,
            report=report,
            disposition='accept'
        )

        # Act
        training_data = dummy_model.get_training_data()
        s1.delete()
        s2.delete()

        # Assert
        assert len(training_data) == 1
        assert training_data[0].__class__ == base.Sentence


class TestDummyModel:
    def test_train_passes(self, dummy_model):
        # Act
        dummy_model.train()  # Has no effect

    def test_test_passes(self, dummy_model):
        # Act
        dummy_model.test()  # Has no effect

    # TODO: Test get_indicators, pick_random_techniques, get_mappings


class TestModelManager:
    @pytest.mark.django_db
    def test___init__loads_tram_model(self):
        # Act
        model_manager = base.ModelManager('tram')

        # Assert
        assert model_manager.model.__class__ == base.TramModel

    @pytest.mark.django_db
    def test__init__loads_dummy_model(self):
        # Act
        model_manager = base.ModelManager('dummy')

        # Assert
        assert model_manager.model.__class__ == base.DummyModel

    def test__init__raises_value_error_on_unknown_model(self):
        # Act / Assert
        with pytest.raises(ValueError):
            base.ModelManager('this-should-raise')

    def test_train_model_doesnt_raise(self):
        # Arrange
        model_manager = base.ModelManager('dummy')

        # Act
        model_manager.train_model()

        # Assert
        # TODO: Something meaningful

    def test_test_model_doesnt_raise(self):
        # Arrange
        model_manager = base.ModelManager('dummy')

        # Act
        model_manager.test_model()

        # Assert
        # TODO: Something meaningful


@pytest.mark.django_db
class TestTramModel:
    def test_train_doesnt_raise(self, simple_training_data, tram_model):
        # Act
        with warnings.catch_warnings():
            # TODO: This makes TONS of warnings that looks like this:
            #       UserWarning: Label not 796 is present in all training examples.
            warnings.simplefilter("ignore")
            tram_model.train()

        # Assert
        # n/a - If .train() doesn't raise, test passes

    def test_test_raises_notimplemented(self, tram_model):
        # Act
        with pytest.raises(NotImplementedError):
            tram_model.test()

    def test_get_mappings_returns_mappings(self, simple_training_data, tram_model):
        # Arrange
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tram_model.train()

        # Act
        mappings = tram_model.get_mappings('This is a test sentence')

        # Assert
        assert mappings.__class__ == list
