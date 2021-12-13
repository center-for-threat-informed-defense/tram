import json
import time
import re

from django.core.files import File
from django.core.management.base import BaseCommand


from tram.ml import base
import tram.models as db_models
from django.core import serializers
from tram.models import Indicator


RUN = 'run'


class Command(BaseCommand):
    help = 'Ioc parsing commands'

    def add_arguments(self, parser):
        sp = parser.add_subparsers(title='subcommands',
                                   dest='subcommand',
                                   required=True)
        sp_run = sp.add_parser(RUN, help='Run ioc parsing')
        sp_run.add_argument('--file', action="store_true",
                             help='If flag is specific will output data to data/ioc-output.json')

    def handle(self, *args, **options):
        subcommand = options['subcommand']

        if subcommand == RUN:
            iocs = {}
            iocs['ipv4'] = re.compile(r'\b(?:(?:1\d\d|2[0-5][0-5]|2[0-4]\d|0?[1-9]\d|0?0?\d)\.){3}(?:1\d\d|2[0-5][0-5]|2[0-4]\d|0?[1-9]\d|0?0?\d)\b')
            iocs['domain'] = re.compile(r'\b([a-z0-9]+(-[a-z0-9]+)*\[?\.\]?)+[a-z]{2,}\b')
            iocs['fqdn'] = re.compile(r'\b(?:[a-z0-9]+(?:-[a-z0-9]+)*\[?\.\]?)+[a-z]{2,}\b')
            iocs['md5'] = re.compile(r'\b(?:[a-fA-F0-9]){32}\b')
            iocs['regkey'] = re.compile(r'\b(?:HKEY_CURRENT_USER\\|SOFTWARE\\|HKEY_LOCAL_MACHINE\\|HKLM\\|HKCR\\|HKCU\\)(?:[A-Z][a-zA-Z]*[\ ]?\\*)*\b')
            iocs['email'] = re.compile(r'\b[\[\]a-zA-Z0-9_.+-]+@[\[\]a-zA-Z0-9-]+\.[\[\]a-zA-Z0-9-.]+\b')
            iocs['filepath_linux'] = re.compile(r'(\/.*?\.[\w:]+[^\s]+)')
            iocs['filepath_windows'] = re.compile(r'(C:\\.*?\.[\w:]+)')
            iocs['cve'] = re.compile(r'\bCVE-[12][0-9]{3}-[0-9]+\b')
            iocs['mac_address'] = re.compile(r'(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})')
            iocs['sha1'] = re.compile(r'b[a-fA-F0-9]{40}\b')
            iocs['sha256'] = re.compile(r'\b[A-Fa-f0-9]{64}\b')
            iocs['sha512'] = re.compile(r'\b[a-fA-f0-9]{128}\b')

            reports = db_models.Report.objects.all()
            for report in reports:
                text = report.text
                for key in iocs:
                    for ioc in iocs[key].findall(text):
                        i = Indicator(report=report,indicator_type=key,value=ioc)
                        i.save()
            IOC_json = serializers.serialize("json",Indicator.objects.all())
            if options['file']:
                with open('data/ioc-output.json','w') as f:
                    f.write(IOC_json)
            else:
                print(IOC_json)
