import json

from django.conf import settings
from django.core.management.base import BaseCommand

from bs4 import BeautifulSoup

import requests
import pdfplumber
from io import BytesIO

from tram.models import AttackTechnique, Mapping, Sentence, Report

LOAD = 'load'


class Command(BaseCommand):
    help = 'Machine learning pipeline commands'

    def add_arguments(self, parser):
        sp = parser.add_subparsers(title='subcommands',
                                   dest='subcommand',
                                   required=True)
        sp_load = sp.add_parser(LOAD, help='Load OTX Data into the Database')

    def load_otx_data(self, filepath):
        with open(filepath, 'r') as f:
            OTX = json.load(f)

        for i in OTX:
            if(len(i['references']) == 0):
                continue
            url = i["references"][0]
            if(url == '' or 'http' not in url or url == 'https://blog.netlab.360.com/blackrota-an-obfuscated-backdoor-written-in-go-en/'):
                continue
            try:
                print("Getting url: {}".format(url))
                text = ''
                r = requests.get(url, stream=True)
                pdf = pdfplumber.open(BytesIO(r.content))
                print(pdf)
                for page in pdf.pages:
                    text += page.extract_text()
                # X.append(text)
                # y.append(i['attack_ids'])
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

                for t in text:
                    if t.parent.name not in blacklist and t != '\n' and '<' not in t and '>' not in t:
                        output += '{} '.format(t)

                r = Report()
                s = Sentence()

                r.ml_model = 'fullreport'
                s.text = output
                r.text = output
                s.report = r
                r.save()
                s.save()
                for id in i['attack_ids']:
                    try:
                        technique = AttackTechnique.objects.get(attack_id=id)
                        m = Mapping()
                        m.attack_technique = technique
                        m.report = r
                        m.sentence = s
                        m.confidence = 100
                        m.save()
                    except:
                        print("Technique non existent, passing")
                        continue

    def handle(self, *args, **options):
        subcommand = options['subcommand']

        if subcommand == LOAD:
            self.load_otx_data(settings.DATA_DIRECTORY / 'training/otx-training-data.json')
        elif subcommand == CLEAR:
            self.clear_otx_data()
