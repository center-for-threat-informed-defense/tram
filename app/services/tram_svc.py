import asyncio
import feedparser
import requests
from app.objects.c_report import Report
import htmldate
import pandas as pd

from app.utility.base_service import BaseService
from app.objects.secondclass.c_match import Match


class TramService(BaseService):

    def __init__(self):
        self.log = self.add_service('tram_svc', self)
        self.data_svc = self.get_service('data_svc')

    async def get_reports(self):
        return [r.display for r in await self.data_svc.locate('reports')]

    async def get_search_terms(self):
        return [s.display for s in await self.data_svc.locate('search')]

    async def get_ttps(self, reports):
        match_report = lambda report : [[m['search']['name'], m['search']['code'], m['confidence']] for m in report['matches']]
        matches = []
        for report in reports:
             matches_m = match_report(report)
             matches_s = [match_report(i) for i in report['sentences']]
             matches.append(matches_m + sum(matches_s, []))
        matches = sum(matches, [])
        if not matches:
            return []

        df = pd.DataFrame(matches)
        df = df.groupby([0,1], as_index=False).agg({2:['mean', 'size']})
        df.columns = ['name', 'code', 'confidence', 'occurrences']
        df = df.sort_values(['occurrences', 'confidence'], ascending=False).reset_index(drop=True)

        return df.to_json(orient="records")

    async def get_all_ttps(self):
        reports = await self.get_reports()
        return await self.get_ttps(reports)

    async def get_current_ttps(self):
        reports = await self.get_reports()
        current = [r for r in reports if r['status'] != 'COMPLETED']
        return await self.get_ttps(current)

    async def get_past_ttps(self):
        reports = await self.get_reports()
        past = [r for r in reports if r['status'] == 'COMPLETED']
        return await self.get_ttps(past)

    async def export_ttps(self,ttp):
        if(ttp == 'all'):
            return await self.export_all_ttps()
        elif(ttp == 'curr'):
            return await self.export_current_ttps()
        elif(ttp == 'past'):
            return await self.export_past_ttps()
        elif(ttp == ''):
            return await self.export_all_ttps()
            
    async def export_all_ttps(self):
        return await self.get_all_ttps()

    async def export_current_ttps(self):
        return await self.get_current_ttps()
    
    async def export_past_ttps(self):
        return await self.get_past_ttps()

    async def create_report(self, new_report):
        if new_report.url:
            new_report.file_date = htmldate.find_date(new_report.url)
        exists = await self.data_svc.locate('reports', dict(id=new_report.id))
        report = await self.data_svc.store(new_report)
        if not exists:
            asyncio.get_event_loop().create_task(self.get_service('machine_svc').learn(report))
    
    async def reassess_report(self,id):
        report = await self.data_svc.locate('reports', dict(id=id))
        new_report = report[0]
        new_report.status = 'QUEUE'
        new_report.matches = []
        asyncio.get_event_loop().create_task(self.get_service('machine_svc').learn(new_report))

    async def retrain_model(self, model_name):
        model = await self.data_svc.locate('model', dict(name=model_name))
        asyncio.get_event_loop().create_task(self.get_service('machine_svc').retrain(model))

    async def export_report(self, report_id, type):
        reports = await self.data_svc.locate('reports', dict(id=report_id))
        return reports[0].export(type)

    async def update_match(self, report_id, match_id, accepted):
        for report in await self.data_svc.locate('reports', dict(id=report_id)):
            for sent in report.sentences:
                for match in [m for m in sent.matches if m.id == match_id]:
                    match.accepted = bool(accepted)
                    self.log.debug('[%s] accepted changed to %s' % (match_id, accepted))

    async def delete_report(self, data):
        await self.data_svc.remove('reports', data)

    async def add_user_match(self, match_desc, sentenceID):
        for s in await self.data_svc.locate('search'):
            if match_desc == s.description:
                search_info = s
                break
        sentenceIDArr = sentenceID.split(",")
        orig_sentences = []
        for sentence in sentenceIDArr:
            for r in await self.data_svc.locate('reports'):
                for sent in r.sentences:
                    if sent.id == sentence:
                        orig_sentences.append(sent)
                        currReport = r
                        break
        for sentObj in orig_sentences:
            sentObj.matches.append(Match(search=search_info, manual=True))
        return currReport.display

    async def delete_match(self,match_desc,sentenceID):
        match_desc = " ".join(match_desc.strip().split(' '))
        for s in await self.data_svc.locate('search'):
            if match_desc == s.description:
                search_info = s
                break
        for r in await self.data_svc.locate('reports'):
            for sent in r.sentences:
                if sent.id == sentenceID:
                    for i in range(len(sent.matches)):
                        if sent.matches[i].search == search_info:
                            del sent.matches[i]
                            await self.data_svc.store(r)
                            break
                            

    async def pull_rss_feed(self, rss_url):
        feed = feedparser.parse(rss_url['url'])
        for entry in feed.entries:
            url = entry.link
            new_report = Report(url=url)
            await self.create_report(new_report)

    async def pull_attack_refs(self):
        r = requests.get('https://github.com/mitre/cti/raw/master/enterprise-attack/enterprise-attack.json',
                         verify=False)
        mobile_attack = requests.get('https://github.com/mitre/cti/raw/master/mobile-attack/mobile-attack.json',
                                     verify=False)
        pre_attack = requests.get('https://github.com/mitre/cti/raw/master/pre-attack/pre-attack.json', verify=False)
        all_attack = [r, mobile_attack, pre_attack]
        attack_reports = []
        for matrix in all_attack:
            outer_stix = matrix.json().get('objects')
            for obj in outer_stix:
                try:
                    inner_stix = obj.get('external_references')
                    for reference in inner_stix:
                        source_name = reference.get('source_name')
                        urlLink = reference.get('url')
                        report = {'source_name': source_name, 'url': urlLink}
                        if (report['source_name'] != 'mitre-attack' and (report['source_name'] != 'mitre-pre-attack')
                                and (report['source_name'] != 'mitre-mobile-attack') and report['url']):
                            attack_reports.append(report)
                except:
                    self.log.error('Failed to pull report/No external references found')
        return attack_reports
