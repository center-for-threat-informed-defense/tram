from django.core.files.base import File
import pytest

from tram.ml import base
import tram.models as db_models


@pytest.fixture
def dummy_model():
    return base.DummyModel()


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
        # doc.delete()

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
        # doc.delete()

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
        # job.delete()
        # doc.delete()

        # Assert
        assert report.name is not None
        assert report.text is not None
        assert len(report.sentences) > 0
        assert len(report.indicators) > 0

    @pytest.mark.skip('Function not implemented')
    @pytest.mark.usefixtures('load_attack_data')
    def test_get_training_data_succeeds(self, dummy_model):
        # Arrange
        with open('tests/data/AA20-302A.docx', 'rb') as f:
            doc = db_models.Document(docfile=File(f))
            doc.save()
        rpt = db_models.Report(name='test report',
                               document=doc,
                               text='test text',
                               ml_model='NoModel'
                               )
        rpt.save()

        # Local aliases
        create_sentence = db_models.Sentence.objects.create
        create_mapping = db_models.Mapping.objects.create

        s1 = create_sentence(text='sentence1', order=0, document=doc)
        s2 = create_sentence(text='sentence2', order=1, document=doc)
        s3 = create_sentence(text='sentence3', order=2, document=doc)
        s4 = create_sentence(text='sentence4', order=3, document=doc)
        s5 = create_sentence(text='sentence5', order=4, document=doc)

        t1 = db_models.AttackTechnique.objects.get(attack_id='T1327')
        t2 = db_models.AttackTechnique.objects.get(attack_id='T1497.003')
        t3 = db_models.AttackTechnique.objects.get(attack_id='T1579')

        # Sentence 1 has no mapping
        create_mapping(report=rpt, sentence=s1, confidence=1.0, attack_technique=None, disposition='accept')

        # Sentence 2 has 1 mapping
        create_mapping(report=rpt, sentence=s2, confidence=2.0, attack_technique=t1, disposition='accept')

        # Sentence 3 has 2 mappings
        create_mapping(report=rpt, sentence=s3, confidence=33.3, attack_technique=t2, disposition='accept')
        create_mapping(report=rpt, sentence=s3, confidence=33.4, attack_technique=t3, disposition='accept')

        # Sentence 4 has 3 mappings, 2 accepted and one rejected
        create_mapping(report=rpt, sentence=s4, confidence=99.9, attack_technique=t1, disposition='accept')
        create_mapping(report=rpt, sentence=s4, confidence=99.9, attack_technique=t2, disposition='accept')
        create_mapping(report=rpt, sentence=s4, confidence=99.9, attack_technique=t3, disposition='reject')

        # Sentence 5 has 2 mappings, both rejected
        create_mapping(report=rpt, sentence=s5, confidence=50.0, attack_technique=t1, disposition='reject')
        create_mapping(report=rpt, sentence=s5, confidence=50.0, attack_technique=t3, disposition='reject')

        # Act
        sentences = dummy_model.get_training_data()

        # Assert
        assert len(sentences) == 3  # There should be 3 sentence objects
        for sentence in sentences:
            assert sentence.__class__ == base.Sentence
            for mapping in sentence.mappings:
                assert mapping.__class__ == base.Mapping
                assert mapping.confidence is not None
                assert mapping.confidence >= 0.0
                assert mapping.confidence <= 100.0
                assert mapping.attack_technique is not None


class TestDummyModel:
    def test_train_passes(self, dummy_model):
        # Act
        dummy_model.train()  # Has no effect

    def test_test_passes(self, dummy_model):
        # Act
        dummy_model.test()  # Has no effect

    # TODO: Test get_indicators, pick_random_techniques, get_mappings


class TestModelManager:
    def test___init__loads_tram_model(self):
        # Act
        model_manager = base.ModelManager('tram')

        # Assert
        assert model_manager.model.__class__ == base.TramModel

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
