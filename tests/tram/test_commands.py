import pytest
from django.core.management import call_command
from django.core.management.base import CommandError

from tram.management.commands import attackdata, pipeline
from tram.ml import base
from tram.models import AttackObject


@pytest.mark.django_db
class TestPipeline:
    def test_add_calls_create_from_file(self, mocker):
        # Arrange
        mock_create = mocker.patch("tram.models.DocumentProcessingJob.create_from_file")
        filepath = "tests/data/simple-test.docx"

        # Act
        call_command("pipeline", pipeline.ADD, file=filepath)

        # Assert
        assert mock_create.called_once()

    @pytest.mark.parametrize(
        "subcommand,to_mock",
        [
            (pipeline.RUN, "run_model"),
            (pipeline.TRAIN, "train_model"),
        ],
    )
    def test_subcommand_calls_correct_function(self, mocker, subcommand, to_mock):
        # Arrange
        mocked_func = mocker.patch.object(base.ModelManager, to_mock, return_value=None)

        # Act
        call_command("pipeline", subcommand, model="dummy")

        # Assert
        assert mocked_func.called_once()

    def test_incorrect_subcommand_raises_commanderror(self):
        # Act / Assert
        with pytest.raises(CommandError):
            call_command("pipeline", "incorrect-subcommand")

    @pytest.mark.django_db
    def test_run_succeeds(self):
        # Act
        call_command("pipeline", pipeline.RUN)

        # Assert
        pass


@pytest.mark.django_db
class TestAttackData:
    def test_load_succeeds(self):
        # Arrange
        expected_object_count = 1461

        # Act
        call_command("attackdata", attackdata.LOAD)
        object_count = AttackObject.objects.all().count()

        # Assert
        assert object_count == expected_object_count

    def test_clear_succeeds(self):
        # Arrange
        expected_techniques = 0

        # Act
        call_command("attackdata", attackdata.LOAD)
        call_command("attackdata", attackdata.CLEAR)
        techniques = AttackObject.objects.all().count()

        # Assert
        assert techniques == expected_techniques

    def test_incorrect_subcommand_raises_commanderror(self):
        # Act / Assert
        with pytest.raises(CommandError):
            call_command("attackdata", "incorrect-subcommand")
