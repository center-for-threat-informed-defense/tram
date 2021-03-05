import logging
import sys
from aiohttp import web
from aiohttp_jinja2 import template, render_template
from app.objects import Report
from app.utility.base_world import BaseWorld
import os.path, time
import json

""" THIS LINE IS TO AUTO-HOOK THE MITRE CALDERA FRAMEWORK, IF APPLICABLE """
sys.path.append('plugins/tram')


class RestApi(BaseWorld):

    def __init__(self, services):
        self.log = logging.getLogger('rest_api')
        self.app_svc = services.get('app_svc')
        self.tram_svc = services.get('tram_svc')
        self.file_svc = services.get('file_svc')
        self.auth_svc = services.get('auth_svc')

    async def enable(self):
        self.app_svc.application.router.add_static('/static', 'static/', append_version=True)
        self.app_svc.application.router.add_route('*', '/', self.landing)
        self.app_svc.application.router.add_route('*', '/enter', self.validate_login)
        self.app_svc.application.router.add_route('*', '/logout', self.logout)
        self.app_svc.application.router.add_route('GET', '/login', self.login)
        self.app_svc.application.router.add_route('*', '/tram/api', self.rest_core)
        self.app_svc.application.router.add_route('POST', '/upload/report', self.upload_file)
        self.app_svc.application.router.add_route('GET', '/plugin/tram/gui', self.splash)
        self.app_svc.application.router.add_route('*', '/report/{id}', self.open_page)
        self.app_svc.application.router.add_route('*', '/analytics', self.show_analytics)

    """ BOILERPLATE """

    @template('login.html', status=401)
    async def login(self, request):
        return dict()

    async def validate_login(self, request):
        return await self.auth_svc.login_user(request)

    async def get_user_list(self):
        return list(self.auth_svc.user_map)

    @template('login.html')
    async def logout(self, request):
        await self.auth_svc.logout_user(request)

    async def landing(self, request):
        access = await self.auth_svc.get_permissions(request)
        if not access:
            return render_template('login.html', request, dict())
        return render_template('index.html', request, dict())

    @template('tram.html')
    async def splash(self, request):
        return dict()

    @template('report.html')
    async def open_page(self, request):
        report_id = request.url.name
        reports = await self.tram_svc.data_svc.locate('reports', dict(id=report_id))
        r = reports[0].display
        return {'report': r}

    @template('analytics.html')
    async def show_analytics(self, request):
        ttps_str = await self.tram_svc.get_all_ttps()
        if ttps_str:
            ttps_dict = json.loads(ttps_str)
            return {'content': ttps_dict}
        else:
            return dict()

    """ API ENDPOINTS """

    async def rest_core(self, request):
        try:
            options = dict(
                DELETE=dict(
                  reports=lambda d: self.tram_svc.delete_report(d),
                ),
                GET=dict(
                    reports=lambda: self.tram_svc.get_reports(),
                    search=lambda: self.tram_svc.get_search_terms(),
                    users=lambda: self.get_user_list(),
                    all_ttps=lambda: self.tram_svc.get_all_ttps(),
                    curr_ttps=lambda: self.tram_svc.get_current_ttps(),
                    past_ttps=lambda: self.tram_svc.get_past_ttps(),
                ),
                POST=dict(
                    reports=lambda d: self.tram_svc.create_report(Report(**d)),
                    reassess=lambda d: self.tram_svc.reassess_report(**d),
                    export=lambda d: self.tram_svc.export_report(**d),
                    exportTTP=lambda d: self.tram_svc.export_ttps(**d),
                    match=lambda d: self.tram_svc.update_match(**d),
                    retrain=lambda d: self.tram_svc.retrain_model(**d),
                    addMatch=lambda d: self.tram_svc.add_user_match(**d),
                    attack=lambda d: self.tram_svc.pull_attack_refs(),
                    rss=lambda d: self.tram_svc.pull_rss_feed(dict(**d)),
                    deleteMatch=lambda d: self.tram_svc.delete_match(**d),
                )
            )
            if request.method == 'GET':
                return web.json_response(await options[request.method][request.rel_url.query['action']]())
            data = dict(await request.json())
            action = data.pop('action')
            return web.json_response(await options[request.method][action](data))
        except Exception as e:
            self.log.error(repr(e), exc_info=True)

    async def upload_file(self, request):
        dir_name = request.headers.get('Directory', None)
        f = await self.file_svc.save_multipart_file_upload(request, dir_name)
        file_date = request.headers.get('file_date')
        await self.tram_svc.create_report(Report(file=f, file_date=file_date))
        return web.json_response(f)
