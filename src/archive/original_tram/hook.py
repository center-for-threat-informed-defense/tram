from app.utility.base_world import BaseWorld
from plugins.tram.app.api.rest import RestApi
from plugins.tram.app.services.machine_svc import MachineService
from plugins.tram.app.services.tram_svc import TramService

name = 'Tram'
description = 'A plugin for the TRAM threat intelligence project'
address = '/plugin/tram/gui'
access = BaseWorld.Access.RED


async def enable(services):
    await services.get('data_svc').apply('reports')
    await services.get('data_svc').apply('search')

    rest_api = RestApi(
        services=dict(data_svc=services.get('data_svc'), app_svc=services.get('app_svc'),
                      file_svc=services.get('file_svc'), machine_svc=MachineService(), tram_svc=TramService())
    )
    services.get('app_svc').application.router.add_static('/tram', 'plugins/tram/static/', append_version=True)
    await rest_api.enable()
