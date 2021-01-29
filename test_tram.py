from app.models.regex import Model as RegexModel
from app.services.app_svc import AppService
from tram import Tram

import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_data_svc():
    data_svc = Mock()

    ram = dict(search=[], reports=[])
    data_svc.ram = ram

    async def store (c_object):
        if c_object.__class__.__name__ == 'Search':
            ram['search'].append(c_object)
        elif c_object.__class__.__name__ == 'Report':
            ram['reports'].append(c_object)
    data_svc.store = store

    async def locate (object_name, criteria=None):
        return [obj for obj in ram[object_name] if not criteria or
                obj.__getattribute__(list(criteria.keys())[0]) == list(criteria.values())[0]]
    data_svc.locate = locate

    return data_svc


@pytest.fixture
async def mock_app_svc(loop, mock_data_svc):
    app_svc = AppService(application=None)
    app_svc.data_svc = mock_data_svc
    await app_svc.load_techniques()


@pytest.fixture
def mock_regex_model(mock_app_svc, mock_data_svc):
    model = RegexModel()
    model._services['data_svc'] = mock_data_svc
    return model


@pytest.fixture
def mock_tram_library(loop, mock_data_svc):
    t = Tram()
    t.data_svc = mock_data_svc
    return t


@pytest.fixture
def mock_report():
    m = Mock(sentences=[], completed_models=0)
    m.export = lambda x : m.__dict__
    return m

@pytest.mark.skip(reason='The test concern is unclear')
async def test_load_techniques(mock_app_svc, mock_data_svc):
    assert len(mock_data_svc.ram['search']) == 847

@pytest.mark.skip(reason='The test concern is unclear')
async def test_mobile_techniques(mock_regex_model, mock_report):
    tokens = ['This is a sentence T1453 .', 'This is a sentence T1401 .', 'This is a sentence T1435 .',
              'This is a sentence T1433 .', 'This is a sentence T1432 .', 'This is a sentence T1517 .',
              'This is a sentence T1413 .', 'This is a sentence T1409 .', 'This is a sentence T1438 .',
              'This is a sentence T1416 .', 'This is a sentence T1402 .', 'This is a sentence T1418 .',
              'This is a sentence T1427 .', 'This is a sentence T1429 .', 'This is a sentence T1512 .',
              'This is a sentence T1414 .', 'This is a sentence T1412 .', 'This is a sentence T1510 .',
              'This is a sentence T1436 .', 'This is a sentence T1532 .', 'This is a sentence T1471 .',
              'This is a sentence T1533 .', 'This is a sentence T1447 .', 'This is a sentence T1475 .',
              'This is a sentence T1476 .', 'This is a sentence T1446 .', 'This is a sentence T1408 .',
              'This is a sentence T1520 .', 'This is a sentence T1466 .', 'This is a sentence T1407 .',
              'This is a sentence T1456 .', 'This is a sentence T1439 .', 'This is a sentence T1523 .',
              'This is a sentence T1428 .', 'This is a sentence T1404 .', 'This is a sentence T1449 .',
              'This is a sentence T1450 .', 'This is a sentence T1405 .', 'This is a sentence T1458 .',
              'This is a sentence T1477 .', 'This is a sentence T1420 .', 'This is a sentence T1472 .',
              'This is a sentence T1417 .', 'This is a sentence T1516 .', 'This is a sentence T1411 .',
              'This is a sentence T1478 .', 'This is a sentence T1464 .', 'This is a sentence T1430 .',
              'This is a sentence T1461 .', 'This is a sentence T1452 .', 'This is a sentence T1463 .',
              'This is a sentence T1444 .', 'This is a sentence T1403 .', 'This is a sentence T1398 .',
              'This is a sentence T1400 .', 'This is a sentence T1399 .', 'This is a sentence T1507 .',
              'This is a sentence T1423 .', 'This is a sentence T1410 .', 'This is a sentence T1406 .',
              'This is a sentence T1470 .', 'This is a sentence T1448 .', 'This is a sentence T1424 .',
              'This is a sentence T1468 .', 'This is a sentence T1469 .', 'This is a sentence T1467 .',
              'This is a sentence T1465 .', 'This is a sentence T1451 .', 'This is a sentence T1513 .',
              'This is a sentence T1437 .', 'This is a sentence T1521 .', 'This is a sentence T1474 .',
              'This is a sentence T1508 .', 'This is a sentence T1426 .', 'This is a sentence T1422 .',
              'This is a sentence T1421 .', 'This is a sentence T1415 .', 'This is a sentence T1509 .',
              'This is a sentence T1481 .']
    answers = [['collection', 'credential-access', 'impact', 'defense-evasion'], ['persistence'], ['collection'],
               ['collection'],
               ['collection'], ['collection', 'credential-access'], ['collection', 'credential-access'],
               ['collection', 'credential-access'], ['command-and-control', 'exfiltration'], ['credential-access'],
               ['persistence'], ['defense-evasion', 'discovery'], ['lateral-movement'], ['collection'], ['collection'],
               ['collection', 'credential-access'], ['collection', 'credential-access'], ['impact'],
               ['command-and-control', 'exfiltration'], ['exfiltration'], ['impact'], ['collection'], ['impact'],
               ['initial-access'], ['initial-access'], ['impact', 'defense-evasion'], ['defense-evasion'],
               ['command-and-control'], ['network-effects'], ['defense-evasion'], ['initial-access'],
               ['network-effects'],
               ['defense-evasion', 'discovery'], ['lateral-movement'], ['privilege-escalation'], ['network-effects'],
               ['network-effects'], ['credential-access', 'privilege-escalation'], ['initial-access'],
               ['initial-access'],
               ['discovery'], ['impact'], ['collection', 'credential-access'], ['defense-evasion', 'impact'],
               ['credential-access'], ['defense-evasion', 'initial-access'], ['network-effects'],
               ['collection', 'discovery'],
               ['initial-access'], ['impact'], ['network-effects'], ['initial-access'], ['persistence'],
               ['defense-evasion', 'persistence'], ['defense-evasion', 'persistence', 'impact'],
               ['defense-evasion', 'persistence'], ['collection'], ['discovery'], ['collection', 'credential-access'],
               ['defense-evasion'], ['remote-service-effects'], ['impact'], ['discovery'], ['remote-service-effects'],
               ['remote-service-effects'], ['network-effects'], ['network-effects'], ['network-effects'],
               ['collection'],
               ['command-and-control', 'exfiltration'], ['command-and-control'], ['initial-access'],
               ['defense-evasion'],
               ['discovery'], ['discovery'], ['discovery'], ['credential-access'], ['command-and-control'],
               ['command-and-control']]

    await mock_regex_model.learn(mock_report, tokens)

    matches = [[match.search.name for match in match_list] for match_list in
               [sen.matches for sen in mock_report.sentences]]
    assert matches == answers


async def test_pre_techniques(mock_regex_model, mock_report):
    tokens = ['This is a sentence T1266 .', 'This is a sentence T1247 .', 'This is a sentence T1277 .',
              'This is a sentence T1329 .', 'This is a sentence T1307 .', 'This is a sentence T1308 .',
              'This is a sentence T1330 .', 'This is a sentence T1310 .', 'This is a sentence T1332 .',
              'This is a sentence T1275 .', 'This is a sentence T1293 .', 'This is a sentence T1288 .',
              'This is a sentence T1301 .', 'This is a sentence T1287 .', 'This is a sentence T1294 .',
              'This is a sentence T1300 .', 'This is a sentence T1289 .', 'This is a sentence T1297 .',
              'This is a sentence T1303 .', 'This is a sentence T1295 .', 'This is a sentence T1306 .',
              'This is a sentence T1229 .', 'This is a sentence T1236 .', 'This is a sentence T1224 .',
              'This is a sentence T1299 .', 'This is a sentence T1302 .', 'This is a sentence T1296 .',
              'This is a sentence T1298 .', 'This is a sentence T1238 .', 'This is a sentence T1228 .',
              'This is a sentence T1381 .', 'This is a sentence T1386 .', 'This is a sentence T1384 .',
              'This is a sentence T1347 .', 'This is a sentence T1349 .', 'This is a sentence T1341 .',
              'This is a sentence T1328 .', 'This is a sentence T1352 .', 'This is a sentence T1391 .',
              'This is a sentence T1343 .', 'This is a sentence T1321 .', 'This is a sentence T1312 .',
              'This is a sentence T1334 .', 'This is a sentence T1354 .', 'This is a sentence T1388 .',
              'This is a sentence T1254 .', 'This is a sentence T1226 .', 'This is a sentence T1253 .',
              'This is a sentence T1279 .', 'This is a sentence T1268 .', 'This is a sentence T1249 .',
              'This is a sentence T1376 .', 'This is a sentence T1383 .', 'This is a sentence T1339 .',
              'This is a sentence T1345 .', 'This is a sentence T1232 .', 'This is a sentence T1355 .',
              'This is a sentence T1231 .', 'This is a sentence T1374 .', 'This is a sentence T1382 .',
              'This is a sentence T1324 .', 'This is a sentence T1320 .', 'This is a sentence T1380 .',
              'This is a sentence T1230 .', 'This is a sentence T1284 .', 'This is a sentence T1260 .',
              'This is a sentence T1245 .', 'This is a sentence T1285 .', 'This is a sentence T1250 .',
              'This is a sentence T1259 .', 'This is a sentence T1258 .', 'This is a sentence T1243 .',
              'This is a sentence T1242 .', 'This is a sentence T1282 .', 'This is a sentence T1244 .',
              'This is a sentence T1241 .', 'This is a sentence T1227 .', 'This is a sentence T1342 .',
              'This is a sentence T1350 .', 'This is a sentence T1255 .', 'This is a sentence T1379 .',
              'This is a sentence T1394 .', 'This is a sentence T1323 .', 'This is a sentence T1326 .',
              'This is a sentence T1286 .', 'This is a sentence T1311 .', 'This is a sentence T1333 .',
              'This is a sentence T1262 .', 'This is a sentence T1261 .', 'This is a sentence T1377 .',
              'This is a sentence T1325 .', 'This is a sentence T1344 .', 'This is a sentence T1364 .',
              'This is a sentence T1234 .', 'This is a sentence T1365 .', 'This is a sentence T1314 .',
              'This is a sentence T1385 .', 'This is a sentence T1233 .', 'This is a sentence T1280 .',
              'This is a sentence T1272 .', 'This is a sentence T1283 .', 'This is a sentence T1225 .',
              'This is a sentence T1270 .', 'This is a sentence T1248 .', 'This is a sentence T1278 .',
              'This is a sentence T1267 .', 'This is a sentence T1269 .', 'This is a sentence T1271 .',
              'This is a sentence T1348 .', 'This is a sentence T1263 .', 'This is a sentence T1274 .',
              'This is a sentence T1276 .', 'This is a sentence T1246 .', 'This is a sentence T1265 .',
              'This is a sentence T1264 .', 'This is a sentence T1389 .', 'This is a sentence T1256 .',
              'This is a sentence T1336 .', 'This is a sentence T1375 .', 'This is a sentence T1252 .',
              'This is a sentence T1273 .', 'This is a sentence T1257 .', 'This is a sentence T1322 .',
              'This is a sentence T1315 .', 'This is a sentence T1316 .', 'This is a sentence T1390 .',
              'This is a sentence T1309 .', 'This is a sentence T1331 .', 'This is a sentence T1318 .',
              'This is a sentence T1319 .', 'This is a sentence T1313 .', 'This is a sentence T1392 .',
              'This is a sentence T1396 .', 'This is a sentence T1251 .', 'This is a sentence T1281 .',
              'This is a sentence T1346 .', 'This is a sentence T1363 .', 'This is a sentence T1353 .',
              'This is a sentence T1305 .', 'This is a sentence T1335 .', 'This is a sentence T1304 .',
              'This is a sentence T1373 .', 'This is a sentence T1239 .', 'This is a sentence T1235 .',
              'This is a sentence T1351 .', 'This is a sentence T1378 .', 'This is a sentence T1291 .',
              'This is a sentence T1290 .', 'This is a sentence T1358 .', 'This is a sentence T1395 .',
              'This is a sentence T1337 .', 'This is a sentence T1338 .', 'This is a sentence T1317 .',
              'This is a sentence T1340 .', 'This is a sentence T1367 .', 'This is a sentence T1369 .',
              'This is a sentence T1368 .', 'This is a sentence T1397 .', 'This is a sentence T1237 .',
              'This is a sentence T1371 .', 'This is a sentence T1366 .', 'This is a sentence T1240 .',
              'This is a sentence T1393 .', 'This is a sentence T1356 .', 'This is a sentence T1357 .',
              'This is a sentence T1359 .', 'This is a sentence T1360 .', 'This is a sentence T1292 .',
              'This is a sentence T1361 .', 'This is a sentence T1387 .', 'This is a sentence T1372 .',
              'This is a sentence T1370 .', 'This is a sentence T1362 .', 'This is a sentence T1327 .']
    answers = [['people-information-gathering'], ['technical-information-gathering'],
               ['organizational-information-gathering'],
               ['establish-&-maintain-infrastructure'], ['adversary-opsec'], ['adversary-opsec'],
               ['establish-&-maintain-infrastructure'], ['adversary-opsec'], ['establish-&-maintain-infrastructure'],
               ['people-information-gathering'], ['technical-weakness-identification'],
               ['technical-weakness-identification'],
               ['organizational-weakness-identification'], ['technical-weakness-identification'],
               ['technical-weakness-identification'], ['organizational-weakness-identification'],
               ['technical-weakness-identification'], ['people-weakness-identification'],
               ['organizational-weakness-identification'], ['people-weakness-identification'], ['adversary-opsec'],
               ['priority-definition-planning'], ['priority-definition-planning'], ['priority-definition-planning'],
               ['organizational-weakness-identification'], ['organizational-weakness-identification'],
               ['people-weakness-identification'], ['organizational-weakness-identification'],
               ['priority-definition-direction'],
               ['priority-definition-planning'], ['launch'], ['compromise'], ['compromise'], ['build-capabilities'],
               ['build-capabilities'], ['persona-development'], ['establish-&-maintain-infrastructure'],
               ['build-capabilities'],
               ['persona-development'], ['persona-development'], ['adversary-opsec'], ['adversary-opsec'],
               ['establish-&-maintain-infrastructure'], ['build-capabilities'], ['compromise'],
               ['technical-information-gathering'], ['priority-definition-planning'],
               ['technical-information-gathering'],
               ['organizational-information-gathering'], ['people-information-gathering'],
               ['technical-information-gathering'],
               ['launch'], ['compromise'], ['establish-&-maintain-infrastructure'], ['build-capabilities'],
               ['priority-definition-planning'], ['build-capabilities'], ['priority-definition-planning'], ['launch'],
               ['launch'],
               ['adversary-opsec'], ['adversary-opsec'], ['launch'], ['priority-definition-planning'],
               ['organizational-information-gathering'], ['technical-information-gathering'], ['target-selection'],
               ['organizational-information-gathering'], ['technical-information-gathering'],
               ['technical-information-gathering'],
               ['technical-information-gathering'], ['target-selection'], ['target-selection'],
               ['organizational-information-gathering'], ['target-selection'], ['target-selection'],
               ['priority-definition-planning'], ['persona-development'], ['build-capabilities'],
               ['technical-information-gathering'], ['stage-capabilities'], ['stage-capabilities'], ['adversary-opsec'],
               ['establish-&-maintain-infrastructure'], ['organizational-information-gathering'], ['adversary-opsec'],
               ['establish-&-maintain-infrastructure'], ['technical-information-gathering'],
               ['technical-information-gathering'],
               ['launch'], ['adversary-opsec'], ['persona-development'], ['stage-capabilities'],
               ['priority-definition-planning'],
               ['stage-capabilities'], ['adversary-opsec'], ['compromise'], ['priority-definition-planning'],
               ['organizational-information-gathering'], ['people-information-gathering'],
               ['organizational-information-gathering'], ['priority-definition-planning'],
               ['people-information-gathering'],
               ['technical-information-gathering'], ['organizational-information-gathering'],
               ['people-information-gathering'],
               ['people-information-gathering'], ['people-information-gathering'], ['build-capabilities'],
               ['technical-information-gathering'], ['people-information-gathering'],
               ['organizational-information-gathering'],
               ['technical-information-gathering'], ['people-information-gathering'],
               ['technical-information-gathering'],
               ['technical-weakness-identification'], ['technical-information-gathering'],
               ['establish-&-maintain-infrastructure'], ['launch'], ['technical-information-gathering'],
               ['people-information-gathering'], ['technical-information-gathering'], ['adversary-opsec'],
               ['adversary-opsec'],
               ['adversary-opsec'], ['adversary-opsec'], ['adversary-opsec'], ['establish-&-maintain-infrastructure'],
               ['adversary-opsec'], ['adversary-opsec'], ['adversary-opsec'], ['persona-development'],
               ['establish-&-maintain-infrastructure'], ['technical-information-gathering'],
               ['organizational-information-gathering'], ['build-capabilities'], ['stage-capabilities'],
               ['build-capabilities'],
               ['adversary-opsec'], ['establish-&-maintain-infrastructure'], ['adversary-opsec'], ['launch'],
               ['priority-definition-direction'], ['priority-definition-planning'], ['build-capabilities'], ['launch'],
               ['technical-weakness-identification'], ['technical-weakness-identification'], ['test-capabilities'],
               ['launch'],
               ['establish-&-maintain-infrastructure'], ['establish-&-maintain-infrastructure'], ['adversary-opsec'],
               ['establish-&-maintain-infrastructure'], ['launch'], ['launch'], ['launch'],
               ['technical-information-gathering'],
               ['priority-definition-direction'], ['launch'], ['launch'], ['priority-definition-direction'],
               ['test-capabilities'], ['test-capabilities'], ['test-capabilities'], ['test-capabilities'],
               ['test-capabilities'],
               ['technical-weakness-identification'], ['test-capabilities'], ['compromise'], ['launch'], ['launch'],
               ['stage-capabilities'], ['establish-&-maintain-infrastructure']]

    await mock_regex_model.learn(mock_report, tokens)

    matches = [[match.search.name for match in match_list] for match_list in
               [sen.matches for sen in mock_report.sentences]]
    assert matches == answers


def test_library_create_report(mock_tram_library):
    name = 'A journey to Zebrocy land'
    url = 'https://www.welivesecurity.com/2019/05/22/journey-zebrocy-land/'
    report = mock_tram_library.create_report(name = name, url = url)
    assert report['name'] == name
    assert report['url'] == url

    match_answers = [('domain', ('msoffice.', '')), ('domain', ('service-and-action.', '-action')),
                     ('domain', ('out.', '')), ('domain', ('text.', '')), ('fqdn', 'text.txt\n'),
                     ('fqdn', 'service-and-action.php\n'), ('fqdn', 'msoffice.exe\n'), ('fqdn', 'out.txt\n'),
                     ('filepath', 'C:\\ProgramData\\Office\\MS\\msoffice.exe'),
                     ('filepath', 'C:\\ProgramData\\Office\\MS\\text.txt'),
                     ('filepath', 'C:\\ProgramData\\Office\\MS\\out.txt'), ('filepath', '/PSW.Agent.OGE'),
                     ('filepath', '/HackTool.PSWDump.D'), ('filepath', '/TrojanDownloader.Sednit.CMT'),
                     ('filepath', '//45.124.132[.]127/DOVIDNIK - (2018).zip'),
                     ('filepath', '//45.124.132[.]127/action-center/centerforserviceandaction/service-and-action.php')]
    match_report = [(m['search']['name'], m['search']['description']) for m in report['matches']]
    assert sorted(match_report) == sorted(match_answers)

def test_library_queue_reports(mock_tram_library):
    url = 'https://www.welivesecurity.com/2019/05/22/journey-zebrocy-land/'
    ids = mock_tram_library.queue_reports(urls=[url])
    report = mock_tram_library.data_svc.ram['reports'][0]
    assert report.id == ids[0][0]
    assert report.url == url

def test_library_get_report(mock_tram_library, mock_report):
    id = '321'
    mock_report.id = id
    mock_tram_library.data_svc.ram['reports'] = [mock_report]

    report = mock_tram_library.get_report(id)
    assert report['id'] == id

def test_library_get_reports(mock_tram_library, mock_report):
    size = 5
    mock_reports = [mock_report for i in range(size)]
    mock_tram_library.data_svc.ram['reports'] = mock_reports

    reports = mock_tram_library.get_reports()
    assert len(reports) == size

def test_library_status_report(mock_tram_library, mock_report):
    id = '321'
    status = 'queue'
    mock_report.id = id
    mock_report.status = status
    mock_tram_library.data_svc.ram['reports'] = [mock_report]

    result = mock_tram_library.status_report(id)
    assert result == status