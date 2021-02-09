import nltk
import urllib3
import json
import boto3
import re
import pickle

from app.objects.c_search import Search
from app.utility.base_world import BaseWorld
from app.models.base_model import Model as BaseModel

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class AppService(BaseWorld):

    def __init__(self, application):
        self.application = application
        self.log = self.add_service('app_svc', self)
        self.data_svc = self.get_service('data_svc')

    async def teardown(self):
        self.log.debug('[!] shutting down server...good-bye')
        await self._services.get('data_svc').save_state()

    async def verify_dependencies(self):
        try:
            nltk.data.find('tokenizers/punkt')
            self.log.debug('Found punkt')
        except LookupError:
            self.log.warning('Could not find the punkt pack, downloading now')
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
            self.log.debug('Found stopwords')
        except LookupError:
            self.log.warning('Could not find the stopwords pack, downloading now')
            nltk.download('stopwords')

    def verify_data_format(self, annotations, negs):
        def verify_list(l):
            assert type(l) == list
            assert all(isinstance(i, str) for i in l)
        for ann in annotations:
            body = annotations[ann]
            if ann[-6:] == "-multi":
                assert type(body) == dict
                assert sorted(list(body.keys())) == ['sentances', 'technique_names']
                verify_list(body['technique_names'])
                verify_list(body['sentances'])
            else:
                verify_list(body)
        verify_list(negs)

    async def load_training(self,technique_translation):
        self.log.debug("Loading training data...")
        with open('data/all_analyzed_reports.json','r') as f:
            annotations = json.loads(f.read())
        with open("data/negative_data.json",'r') as f:
            negs = json.loads(f.read())
        self.verify_data_format(annotations,negs)
        keys = set([key for key in list(annotations.keys()) if '-multi' not in key])
        labels = []
        sentances = []
        for ann in annotations:
            if ann[-6:] == "-multi":
                tech_names = annotations[ann]['technique_names']
                tech_names = list(set.intersection(set(tech_names), keys))
                for i in annotations[ann]['sentances']:
                    labels.append(tech_names)
                    sentances.append(i)
            else:
                tech_names = [ann] if ann in keys else []
                labels.append(tech_names)
                sentances.extend(annotations[ann])
        for i in range(5694): #negs:
            sentances.append(negs[i])
            labels.append(["NO_TECHNIQUE"])
        for i in range(25): # force model to see any blanks as nothing, just in case errors happen
            sentances.append("")
            labels.append(["NO_TECHNIQUE"])
        for i in range(len(labels)):
            await self.data_svc.store(
                Search(tag='training_data', name=labels[i],
                    code=[technique_translation.get(j) if j != 'NO_TECHNIQUE' else j for j in labels[i]],
                    description=sentances[i])
            )

    async def load_techniques(self):
        self.log.debug("Loading techniques...")
        technique_translation = dict()
        technique_files = ['data/enterprise-attack.json', 'data/mobile-attack.json', 'data/pre-attack.json']
        for technique_file in technique_files:
            with open(technique_file,'r') as f:
                stix_json = json.loads(f.read())
            stix_objects = stix_json.get('objects')
            for stix_object in stix_objects:
                try:
                    if stix_object.get('type') != 'attack-pattern' or stix_object.get('revoked'): continue
                    name = stix_object.get('name')
                    technique_id = [ref.get("external_id") for ref in stix_object.get('external_references')
                                    if 'mitre' in ref.get('source_name')][0]
                    technique_translation[name.lower()] = technique_id
                    tactics = [phase.get('phase_name') for phase in stix_object.get('kill_chain_phases')]
                    for tactic in tactics:
                        await self.data_svc.store(Search(tag='attack', name=tactic, code=technique_id, description=name))
                except:
                    print('Object failed to ingest')
        return(technique_translation)

    async def load_model(self):
        self.log.debug("Loading model...")
        with open('data/base_model-v1.0.0.pkl', 'rb') as f:
            model = pickle.load(f)
        await self.get_service('data_svc').store(model)

    '''
    def get_s3(self,version,extension):
    
        s3 = boto3.resource('s3', verify=False)
        bucket = s3.Bucket('mitre-tram-s3')
        for object in bucket.objects.filter(Prefix = version):
            matches = re.findall(r'(.+\/)*(.*\.%s)' % extension,object.key)
            if(len(matches) > 0):
                bucket.download_file(object.key,'./data/'+matches[0][1])
    '''

    async def load_data(self,version):
        try:
            technique_translation = await self.load_techniques()
            await self.load_training(technique_translation)
        except FileNotFoundError:
            '''
            self.log.debug('Data has not been downloaded yet. Downloading...')
            try:
                self.get_s3(version, 'json')
            except Exception as e:
                self.log.error("Failed to download data")
                self.log.error(e)
                return
            technique_translation = await self.load_techniques()
            await self.load_training(technique_translation)
            '''
            self.log.error('Data has not been downloaded yet, please download the files from github releases')
            return
        except Exception as e:
            self.log.debug('ERROR: {}'.format(e))
        self.log.debug('Finished loading data')

    async def load_models(self,version):
        try:
            await self.load_model()
        except FileNotFoundError:
            self.log.debug('Model has not been downloaded yet. Downloading...')
            try:
                self.get_s3(version, 'pkl')
                await self.load_model()
            except Exception:
                self.log.debug("Model did not download from s3. Training...")
                await BaseModel().train()
                return
        except Exception as e:
            self.log.debug('ERROR: {}'.format(e))
        self.log.debug('Finished loading model')