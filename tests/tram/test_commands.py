from django.core.management import call_command
from django.core.management.base import CommandError
import pytest

from tram.management.commands import pipeline, attackdata, loaddata
from tram.ml import base
from tram.models import AttackTechnique, Sentence, Report, Mapping


class TestPipeline:
    def test_add_calls_create_from_file(self, mocker):
        # Arrange
        mock_create = mocker.patch('tram.models.DocumentProcessingJob.create_from_file')
        filepath = 'tests/data/simple-test.docx'

        # Act
        call_command('pipeline', pipeline.ADD, file=filepath)

        # Assert
        assert mock_create.called_once()

    @pytest.mark.parametrize("subcommand,to_mock", [
        (pipeline.RUN, 'run_model'),
        (pipeline.TRAIN, 'train_model'),
    ])
    def test_subcommand_calls_correct_function(self, mocker, subcommand, to_mock):
        # Arrange
        mocked_func = mocker.patch.object(base.ModelManager, to_mock, return_value=None)

        # Act
        call_command('pipeline', subcommand, model='dummy')

        # Assert
        assert mocked_func.called_once()

    def test_incorrect_subcommand_raises_commanderror(self):
        # Act / Assert
        with pytest.raises(CommandError):
            call_command('pipeline', 'incorrect-subcommand')

    @pytest.mark.django_db
    def test_run_succeeds(self, load_attack_data):
        # Act
        call_command('pipeline', pipeline.RUN)

        # Assert
        pass


@pytest.mark.django_db
class TestLoadData:
    @pytest.mark.django_db
    def test_load_training_data_succeeds(self, load_attack_data):
        # Act
        call_command('loaddata', loaddata.LOAD_TRAINING_DATA)

        # Assert
        assert Sentence.objects.count() == 12588  # Count of sentences data/training/bootstrap-training-data.json

    def test_load_otx_data_succeeds(self, load_attack_data):
        # Act
        call_command('loaddata', loaddata.LOAD_OTXDATA, file='tests/data/otx-test-data.json')

        # Assert
        assert Sentence.objects.count() == 1
        assert Report.objects.filter(ml_model='fullreport').count() == 1
        assert Mapping.objects.count() == 15


@pytest.mark.django_db
class TestAttackData:
    def test_load_succeeds(self):
        # Arrange
        expected_techniques = 797

        # Act
        call_command('attackdata', attackdata.LOAD)
        techniques = AttackTechnique.objects.all().count()

        # Assert
        assert techniques == expected_techniques

    def test_clear_succeeds(self):
        # Arrange
        expected_techniques = 0

        # Act
        call_command('attackdata', attackdata.LOAD)
        call_command('attackdata', attackdata.CLEAR)
        techniques = AttackTechnique.objects.all().count()

        # Assert
        assert techniques == expected_techniques

    def test_incorrect_subcommand_raises_commanderror(self):
        # Act / Assert
        with pytest.raises(CommandError):
            call_command('attackdata', 'incorrect-subcommand')
