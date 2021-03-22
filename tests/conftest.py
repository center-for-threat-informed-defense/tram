import pytest

from tram.management.commands import attackdata


@pytest.fixture
def load_attack_data():
    command = attackdata.Command()
    command.handle(subcommand=attackdata.LOAD)
