import argparse
import asyncio
import logging
import sys
import aiohttp_jinja2
import jinja2

from aiohttp import web

from app.api.rest import RestApi
from app.services.app_svc import AppService
from app.services.auth_service import AuthService
from app.services.data_svc import DataService
from app.services.file_svc import FileService
from app.services.machine_svc import MachineService
from app.services.tram_svc import TramService
from app.utility.base_world import BaseWorld


def setup_logger(level=logging.DEBUG):
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(levelname)-5s (%(filename)s:%(lineno)s %(funcName)s) %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    for logger_name in logging.root.manager.loggerDict.keys():
        if logger_name in ('aiohttp.server', 'asyncio'):
            continue
        else:
            logging.getLogger(logger_name).setLevel(100)


async def build_docs():
    process = await asyncio.create_subprocess_exec('sphinx-build', 'docs/', 'docs/_build/html',
                                                   '-b', 'html', '-c', 'docs/',
                                                   stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    await process.communicate()


async def load_templates():
    aiohttp_jinja2.setup(app_svc.application, loader=jinja2.FileSystemLoader(['templates']))


async def start_server():
    await auth_svc.apply(app_svc.application, BaseWorld.get_config('users'))
    app_svc.application.router.add_static('/docs/', 'docs/_build/html', append_version=True)
    runner = web.AppRunner(app_svc.application)
    await runner.setup()
    await web.TCPSite(runner, BaseWorld.get_config('host'), BaseWorld.get_config('port')).start()


def run_tasks(services):
    version = str(BaseWorld.get_config(prop='version', name='default'))
    loop = asyncio.get_event_loop()
    loop.create_task(build_docs())
    loop.create_task(app_svc.verify_dependencies())
    loop.create_task(load_templates())
    loop.create_task(app_svc.load_data(version))
    loop.create_task(app_svc.load_models(version))
    loop.create_task(data_svc.restore_state())
    loop.run_until_complete(RestApi(services).enable())
    loop.run_until_complete(start_server())
    try:
        logging.debug('All systems ready.')
        logging.info('TRAM is running! App can be accessed at http://localhost:9999')
        loop.run_forever()
    except KeyboardInterrupt:
        loop.run_until_complete(services.get('app_svc').teardown())


if __name__ == '__main__':
    def list_str(values):
        return values.split(',')
    sys.path.append('')
    parser = argparse.ArgumentParser('python server.py')
    parser.add_argument('-l', '--log', dest='logLevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level", default='DEBUG')
    parser.add_argument('--fresh', action='store_true', required=False, default=False,
                        help='Run tram with a fresh instance. All reports/models/etc removed for a clean slate')
    parser.add_argument('-c', '--config', dest='config', required=False,
                        help="Set the base config file (DEFAULT: conf/default.yml)",default='conf/default.yml')
    parser.add_argument('-r', '--regex', dest='regex', required=False,default='conf/regex.yml',
                        help="Set the regex file containing the IOC parsers (DEFAULT: conf/regex.yml)")
    args = parser.parse_args()
    setup_logger(getattr(logging, args.logLevel))
    logging.info('TRAM is starting up, please wait...')
    BaseWorld.apply_config('default', BaseWorld.strip_yml(args.config)[0])
    BaseWorld.apply_config('regex', BaseWorld.strip_yml(args.regex)[0])

    data_svc = DataService()
    file_svc = FileService()
    tram_svc = TramService()
    auth_svc = AuthService()
    machine_svc = MachineService()
    app_svc = AppService(application=web.Application())

    if args.fresh:
        asyncio.get_event_loop().run_until_complete(data_svc.destroy())
    run_tasks(services=app_svc.get_services())
