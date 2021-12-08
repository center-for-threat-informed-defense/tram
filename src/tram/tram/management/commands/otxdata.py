import json

from bs4 import BeautifulSoup

import requests
import pdfplumber
from io import BytesIO

from tram.models import AttackTechnique, Mapping, Sentence, Report
from tram.models import Adversary, AdversaryMapping


class Otxdata():

    def load_otx_data(self, filepath):
        with open(filepath, 'r') as f:
            OTX = json.load(f)

        for i in OTX:
            if(len(i['references']) == 0 or len(i['attack_ids']) == 0):
                continue
            url = i["references"][0]
            if(url == '' or 'http' not in url or
                    url == 'https://blog.netlab.360.com/blackrota-an-obfuscated-backdoor-written-in-go-en/'):
                continue
            try:
                print("Getting url: {}".format(url))
                text = ''
                r = requests.get(url, stream=True)
                pdf = pdfplumber.open(BytesIO(r.content))
                print(pdf)
                for page in pdf.pages:
                    text += page.extract_text()
                if len(text) < 500:
                    continue
            except Exception as e:
                print(e)
                try:
                    r = requests.get(url)
                except Exception as e:
                    print(e)
                    continue
                soup = BeautifulSoup(r.content, 'html.parser')
                text = soup.find_all(text=True)

                output = ''
                blacklist = [
                    '[document]',
                    'noscript',
                    'header',
                    'html',
                    'meta',
                    'head',
                    'input',
                    'script',
                    'style',
                    'img'
                    # there may be more elements you don't want, such as "style", etc.
                ]

                if len(text) < 500:
                    continue

                for t in text:
                    if t.parent.name not in blacklist and t != '\n' and '<' not in t and '>' not in t:
                        output += '{} '.format(t)

            r = Report()
            s = Sentence()

            r.ml_model = 'fullreport'
            s.text = output
            s.disposition = 'Accepted'
            r.text = output
            s.report = r
            r.save()
            s.save()

            for id in i['attack_ids']:
                try:
                    technique = AttackTechnique.objects.get(attack_id=id)
                    m = Mapping(attack_technique=technique, report=r, sentence=s, confidence=99.9)
                    m.save()
                except Exception:
                    print("Technique non existent, adding")
                    technique = AttackTechnique(name=id, attack_id=id, stix_id=id)
                    technique.save()

    def load_otx_groups(self, filepath):
        with open(filepath, 'r') as f:
            OTX = json.load(f)

        for i in OTX['results']:
            if i['adversary'] == None or i['adversary'] == '':
                continue
            else:
                a = Adversary.objects.get_or_create(name=i['adversary'])
                r = Report.objects.get_or_create(text=i['description'],ml_model='adversary')

                m = AdversaryMapping(report=r[0],adversary=a[0])
                m.save()

        