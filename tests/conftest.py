import pytest

from tram import settings
from tram.management.commands import attackdata


@pytest.fixture(scope='session')
def x_temporary_media_root(tmpdir_factory):
    old_media_root = getattr(settings, 'MEDIA_ROOT', None)

    temporary_directory = tmpdir_factory.mktemp('tram-test').strpath
    settings.MEDIA_ROOT = temporary_directory

    yield  # Tests get executed

    if old_media_root:
        settings.MEDIA_ROOT = old_media_root


@pytest.fixture
def load_attack_data():
    command = attackdata.Command()
    command.handle(subcommand=attackdata.LOAD)
