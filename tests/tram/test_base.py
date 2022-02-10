import pytest
from constance import config
from django.contrib.auth.models import User
from django.core.files import File

import tram.models as db_models
from tram.ml import base


@pytest.fixture
def dummy_model():
    return base.DummyModel()


class TestSentence:
    def test_sentence_stores_no_mapping(self):
        # Arrange
        text = "this is text"
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
        attack_id = "T1327"
        expected = "Confidence=95.342000; Attack ID=T1327"

        # Act
        m = base.Mapping(confidence, attack_id)

        assert str(m) == expected


class TestReport:
    def test_report_stores_properties(self):
        # Arrange
        name = "Test report"
        text = "Test report text"
        sentences = [base.Sentence("test sentence text", 0, None)]

        # Act
        rpt = base.Report(name=name, text=text, sentences=sentences)

        # Assert
        assert rpt.name == name
        assert rpt.text == text
        assert rpt.sentences == sentences


@pytest.mark.django_db
class TestModelWithoutAttackData:
    """Tests ml.base.Model via DummyModel, without the load_attack_data fixture"""

    def test_get_attack_techniques_raises_if_not_initialized(self, dummy_model):
        # Act / Assert
        with pytest.raises(ValueError):
            dummy_model.get_attack_object_ids()


@pytest.mark.django_db
@pytest.mark.usefixtures("load_attack_data")
class TestSkLearnModel:
    """Tests ml.base.SKLearnModel via DummyModel"""

    def test__sentence_tokenize_works_for_paragraph(self, dummy_model):
        # Arrange
        paragraph = """Hello. My name is test. I write sentences. Tokenize, tokenize, tokenize!
                    When will this entralling text stop, praytell? Nobody knows; the author can't stop.
                    """
        expected = [
            "Hello.",
            "My name is test.",
            "I write sentences.",
            "Tokenize, tokenize, tokenize!",
            "When will this entralling text stop, praytell?",
            "Nobody knows; the author can't stop.",
        ]

        # Act
        sentences = dummy_model._sentence_tokenize(paragraph)

        # Assert
        assert expected == sentences

    @pytest.mark.parametrize(
        "filepath,expected",
        [
            ("tests/data/AA20-302A.pdf", "GLEMALT With a Ransomware Chaser"),
            (
                "tests/data/AA20-302A.docx",
                "Page 22 of 22 | Product ID: AA20-302A  TLP:WHITE",
            ),
            (
                "tests/data/AA20-302A.html",
                "CISA is part of the Department of Homeland Security",
            ),
        ],
    )
    def test__extract_text_succeeds(self, dummy_model, filepath, expected):
        # Arrange
        with open(filepath, "rb") as f:
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
        with open("tests/data/unknown-extension.fizzbuzz", "rb") as f:
            doc = db_models.Document(docfile=File(f))
            doc.save()

        # Act / Assert
        with pytest.raises(ValueError):
            dummy_model._extract_text(doc)

        # Cleanup
        doc.delete()

    def test_get_report_name_succeeds(self, dummy_model):
        # Arrange
        expected = "Report for AA20-302A"
        with open("tests/data/AA20-302A.docx", "rb") as f:
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

    def test_get_attack_objects_succeeds_after_initialization(self, dummy_model):
        # Act
        objects = dummy_model.get_attack_object_ids()

        # Assert
        assert "T1327" in objects  # Ensures mitre-pre-attack is available
        assert "T1497.003" in objects  # Ensures mitre-attack is available
        assert "T1579" in objects  # Ensures mitre-mobile-attack is available

    def test_disk_round_trip_succeeds(self, dummy_model, tmpdir):
        # Arrange
        filepath = (tmpdir + "dummy_model.pkl").strpath

        # Act
        dummy_model.get_attack_object_ids()  # Change the state of the DummyModel
        dummy_model.save_to_file(filepath)

        dummy_model_2 = base.DummyModel.load_from_file(filepath)

        # Assert
        assert dummy_model.__class__ == dummy_model_2.__class__
        assert (
            dummy_model.get_attack_object_ids() == dummy_model_2.get_attack_object_ids()
        )

    def test_no_data_get_training_data_succeeds(self, dummy_model):
        # Act
        X, y = dummy_model.get_training_data()

        # Assert
        assert len(X) == 0
        assert len(y) == 0

    def test_get_training_data_returns_only_accepted_sentences(
        self, dummy_model, report
    ):
        # Arrange
        s1 = db_models.Sentence.objects.create(
            text="sentence1",
            order=0,
            document=report.document,
            report=report,
            disposition=None,
        )
        s2 = db_models.Sentence.objects.create(
            text="sentence 2",
            order=1,
            document=report.document,
            report=report,
            disposition="accept",
        )
        m1 = db_models.Mapping.objects.create(
            report=report,
            sentence=s2,
            attack_object=db_models.AttackObject.objects.get(attack_id="T1548"),
            confidence=100.0,
        )
        config.ML_ACCEPT_THRESHOLD = 0  # Set the threshold to 0 for this test

        # Act
        X, y = dummy_model.get_training_data()
        s1.delete()
        s2.delete()
        m1.delete()

        # Assert
        assert len(X) == 1
        assert len(y) == 1

    def test_non_sklearn_pipeline_raises(self):
        # Arrange
        class NonSKLearnPipeline(base.SKLearnModel):
            def get_model(self):
                return "This is not an sklearn.pipeline.Pipeline instance"

        # Act
        with pytest.raises(TypeError):
            NonSKLearnPipeline()


@pytest.mark.django_db
@pytest.mark.usefixtures("load_attack_data", "load_small_training_data")
class TestsThatNeedTrainingData:
    """
    Loading the training data is a large time cost, so this groups tests together that use
    the training data, even if it doesn't follow the class structure.
    TODO: Training data is loaded for every test in this class. This does
          not provide the efficiency I had assumed.
    """

    """
    ----- Begin ModelManager Tests -----
    """

    def test_modelmanager__init__loads_dummy_model(self):
        # Act
        model_manager = base.ModelManager("dummy")

        # Assert
        assert model_manager.model.__class__ == base.DummyModel

    def test_modelmanager__init__raises_value_error_on_unknown_model(self):
        # Act / Assert
        with pytest.raises(ValueError):
            base.ModelManager("this-should-raise")

    def test_modelmanager_train_model_doesnt_raise(self):
        # Arrange
        model_manager = base.ModelManager("dummy")

        # Act
        model_manager.train_model()

        # Assert
        # TODO: Something meaningful

    """
    ----- End ModelManager Tests -----
    """

    def test_get_mappings_returns_mappings(self):
        # Arrange
        dummy_model = base.DummyModel()
        dummy_model.train()
        dummy_model.test()
        config.ML_CONFIDENCE_THRESHOLD = 0

        # Act
        mappings = dummy_model.get_mappings("test sentence")

        # Assert
        for mapping in mappings:
            assert isinstance(mapping, base.Mapping)

    def test_process_job_produces_valid_report(self):
        # Arrange
        with open("tests/data/AA20-302A.docx", "rb") as f:
            doc = db_models.Document(docfile=File(f))
            doc.save()
        job = db_models.DocumentProcessingJob(document=doc)
        job.save()
        dummy_model = base.DummyModel()
        dummy_model.train()
        dummy_model.test()

        # Act
        report = dummy_model.process_job(job)

        # Cleanup
        job.delete()
        doc.delete()

        # Assert
        assert report.name is not None
        assert report.text is not None
        assert len(report.sentences) > 0

    def test_process_job_handles_image_based_pdf(self):
        """
        Some PDFs can be saved such that the text is stored as images and therefore
        cannot be extracted from the PDF. Windows PDF Printer behaves this way.

        Image-based PDFs cause the processing pipeline to fail. The expected behavior
        is that the job is logged as "status: error".
        """
        # Arrange
        image_pdf = "tests/data/GroupIB_Big_Airline_Heist_APT41.pdf"
        dummy_user = User.objects.get_or_create(username="dummy-user")[0]
        with open(image_pdf, "rb") as f:
            processing_job = db_models.DocumentProcessingJob.create_from_file(File(f), dummy_user)
        job_id = processing_job.id
        model_manager = base.ModelManager("dummy")

        # Act
        model_manager.run_model()
        job_result = db_models.DocumentProcessingJob.objects.get(id=job_id)

        # Assert
        assert job_result.status == "error"
        assert len(job_result.message) > 0

    """
    ----- Begin DummyModel Tests -----
    """

    def test_dummymodel_train_and_test_passes(self, dummy_model):
        # Act
        dummy_model.train()  # Has no effect
        dummy_model.test()  # Has no effect

    """
    ----- End DummyModel Tests -----
    """
