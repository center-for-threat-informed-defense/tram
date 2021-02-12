import networkx as nx
from node2vec import Node2Vec
import numpy as np
import spacy
from tqdm import tqdm
import pickle
import forestci as fci
import logging
import yaml
from enum import Enum
import copy
import glob
import os.path
import pickle
import shutil
import nltk
import urllib3
import json
import boto3
import re
import pickle
import uuid
import pdfplumber

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import sklearn.ensemble as lm
import scipy.stats as st


# Popped out of BaseWorld
def retrieve(collection, id):
    return next((i for i in collection if i.id == id), None)
# End popped out of BaseWorld

# Popped out of Tram
def strip_yml(path):
    with open(path, encoding='utf-8') as file:
        return list(yaml.load_all(file, Loader=yaml.FullLoader))[0]
# End popped out of Tram

class FileParser(object):
    @staticmethod
    def parse_file(f):
        pieces = f.split('.')
        if not len(pieces) > 1:  # file path + extension
            return _parse_text(f)
        if pieces[-1] == 'txt':
            return FileParser._parse_text(f)
        elif pieces[-1] == 'pdf':
            return FileParser._parse_pdf(f)

    @staticmethod
    def _parse_text(f):
        with open(f, 'r') as f:
            return f.read()

    @staticmethod
    def _parse_pdf(f):
        with pdfplumber.open(f) as pdf  :
            return ''.join(page.extract_text() for page in pdf.pages)


class RegexParser(object):

    @staticmethod
    def find(regex, report, blob):
        for m in set([m for m in re.findall(regex['regex'], blob)]):
                report.matches.append(
                    Match(search=Search(tag='ioc', name=regex['name'], description=m, code=regex['code']))
                )


class Tram(object):
    schema = None
    display_schema = None
    load_schema = None

    def __init__(self):
        self.log = logging.getLogger('base_model')  # self.create_logger('base_model')  # self.add_service('base_model', self)
        self.name = 'base model'
        self.model = None
        self.rnc = None
        self.classes = None
        self.tfid = None
        self.count_vec = None
        self.schema = dict(search=[], reports=[])
        self.ram = copy.deepcopy(self.schema)
        self.regex_parser = RegexParser()
        self.regex_expressions = strip_yml('src/pipeline/regex.yml')

    # Functions from AppSvc
    def load_data(self,version):
        try:
            technique_translation = self.load_techniques()
            self.load_training(technique_translation)
        except FileNotFoundError:
            self.log.error('Data has not been downloaded yet, please download the files from github releases')
            return
        except Exception as e:
            self.log.debug('ERROR: {}'.format(e))
        self.log.debug('Finished loading data')

    def load_training(self, technique_translation):
        self.log.debug("Loading training data...")
        with open('data/all_analyzed_reports.json','r') as f:
            annotations = json.loads(f.read())
        with open("data/negative_data.json",'r') as f:
            negs = json.loads(f.read())
        self.verify_data_format(annotations, negs)
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
            self.store(
                Search(tag='training_data', name=labels[i],
                    code=[technique_translation.get(j) if j != 'NO_TECHNIQUE' else j for j in labels[i]],
                    description=sentances[i])
            )

    def load_techniques(self):
        self.log.debug("Loading techniques...")
        technique_translation = dict()
        technique_files = ['data/pre-attack.json']
        #technique_files = ['data/enterprise-attack.json', 'data/mobile-attack.json', 'data/pre-attack.json']
        for technique_file in technique_files:
            with open(technique_file,'r') as f:
                stix_json = json.loads(f.read())
            stix_objects = stix_json.get('objects')
            for stix_object in stix_objects:
                if stix_object.get('type') != 'attack-pattern' or stix_object.get('revoked'): continue
                name = stix_object.get('name')
                technique_id = [ref.get("external_id") for ref in stix_object.get('external_references')
                                if 'mitre' in ref.get('source_name')][0]
                technique_translation[name.lower()] = technique_id
                tactics = [phase.get('phase_name') for phase in stix_object.get('kill_chain_phases')]
                for tactic in tactics:
                    search = Search(tag='attack', name=tactic, code=technique_id, description=name)
                    search.store(self.ram)
                    # self.data_svc_store(Search(tag='attack', name=tactic, code=technique_id, description=name))

        return(technique_translation)

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

    def load_model(self):
        self.log.debug("Loading model...")
        with open('data/ml-models/tram-v1.0.0.pkl', 'rb') as f:
            model = pickle.load(f)
        model.store(self.ram)  # self.data_svc_store(model)

    @property
    def id(self):
        return self.name

    def create_graph_matricies(self, y, classes):
        nodes = []
        for node in range(len(classes)):
            nodes.append(node)

        edges = []
        self.log.debug('Getting edges...')
        for row in tqdm(y):
            for i, row_i in enumerate(row):
                if row_i == 1:
                    for j in range(i, len(row)):
                        if row[j] == 1:
                            edges.append([i, j])
        self.log.debug('Edges complete')

        return nodes, edges

    def embedding_encode(self, y, model):
        out_y = []
        for i in y:
            indices = np.where(i)[0]
            try:
                if indices.size:
                    vecs = [model.wv.get_vector(str(j)) for j in indices]
                    vecs = np.array(vecs).sum(axis=0)
                    out_y.append(vecs / len(indices))
                else:
                    out_y.append(np.zeros(32))
            except:
                self.log.error("Vocab not found error")
        return np.array(out_y)

    def train_embedder(self, y_embed, y, k=5):
        rnc = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        self.log.debug('Fitting...')
        rnc.fit(y_embed, y)
        return rnc

    def embedding_decode(self, y_embed, rnc):
        self.log.debug('Decoding embedding')
        predicted_labels = rnc.predict(y_embed)
        self.log.debug('Finished')
        return predicted_labels

    def extract_X(self, X):
        new_X = self.remove_stops(X)

        count_vec = CountVectorizer(max_features=2500)
        all_counts = count_vec.fit_transform(new_X)
        self.count_vec = count_vec

        tfid = TfidfTransformer()
        ext_X = tfid.fit_transform(all_counts)
        self.tfid = tfid

        X_train = ext_X.toarray()
        return X_train

    def extract_y(self, y):
        binarizer = MultiLabelBinarizer()
        y = self.remove_nones(y)
        Y = binarizer.fit_transform(y)
        self.classes = binarizer.classes_
        nodes, edges  = self.create_graph_matricies(Y, self.classes)
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(nodes)
        nx_graph.add_edges_from(edges)

        N2V = Node2Vec(nx_graph, dimensions=32, walk_length=30, num_walks=300, workers=1)
        n2v = N2V.fit(window=10, min_count=1, batch_words=8)
        return Y, n2v

    def train(self):
        """
        Trains and saves model
        """
        self.log.debug("Training model...")
        search = self.data_svc_locate('search', dict(tag='attack'))
        training_data = self.data_svc_locate('search', dict(tag='training_data'))
        reports = self.data_svc_locate('reports',dict(status=Status.COMPLETED))

        labels_r, items_r = self.parse_reports(reports)
        labels_s, items_s = self.parse_search(search)
        labels_t, items_t = self.parse_training_data(training_data)
        X = items_s + items_t + items_r
        y = labels_s + labels_t + labels_r

        X_train = self.extract_X(X)
        ext_y, n2v = self.extract_y(y)
        new_y = self.embedding_encode(ext_y, n2v)
        self.rnc = self.train_embedder(new_y, ext_y)

        self.model = lm.RandomForestRegressor(n_jobs=-1)
        self.log.debug("base_model: fitting regression model")
        self.model.fit(X_train, new_y)
        self.log.debug("base_model: regression model fit")
        self.inbag = fci.calc_inbag(len(X), self.model)

        self.log.debug("base_model: testing model...")
        test = self.model.predict(X_train)
        lab = self.embedding_decode(test, self.rnc)
        score = f1_score(ext_y, lab, average='weighted')
        self.log.debug("f1 score on training data: {}".format(score))
        self.store(self.ram)  # self.data_svc_store(self)
        # TODO: Provide some kind of application control over this
        with open('data/ml-models/tram-v1.0.0.pkl', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.log.debug("Model trained")

    def store(self, ram):
        if not 'model' in ram.keys():
            existing = None
        else:
            existing = retrieve(ram['model'], self.id)
        if not existing:
            if 'model' in ram.keys():
                ram['model'].append(self)
            else:
                ram['model'] = [self]
            return retrieve(ram['model'], self.id)
        existing.update('name', self.name)
        return existing

    def add_matches(self, full_out, confidence, report, tokens):
        search = self.data_svc_locate('search', dict(tag='attack'))

        for i in range(len(full_out)):
            sen = Sentence(text=tokens[i])
            if len(full_out[i]) > 0:
                for j in full_out[i]:
                    for s in search:
                        if j == s.code:
                            sen.matches.append(Match(model=self.name, search=s, confidence=confidence[i]))
            report.sentences.append(sen)
        report.completed_models += 1

    def calc_confidence(self, X, y):
        # Heavily adapted from https://github.com/scikit-learn-contrib/forest-confidence-interval

        n_trees = self.model.n_estimators
        n_train_samples = self.inbag.shape[0]

        pred = np.array([tree.predict(X) for tree in self.model]).transpose(1, 2, 0)
        pred_mean = np.mean(pred, (0, 1))
        pred_centered = pred - pred_mean

        n_var = np.mean(np.square(self.inbag[0:n_trees]).mean(axis=1).T.view() -
                        np.square(self.inbag[0:n_trees].mean(axis=1)).T.view())
        boot_var = np.square(pred_centered).sum(axis=2) / n_trees
        bias_correction = n_train_samples * n_var * boot_var / n_trees

        variance = np.sum((np.dot(self.inbag - 1, pred_centered.transpose(0, 2, 1)) / n_trees) ** 2, 0)
        variance_unbiased = variance - bias_correction

        confidence_interval = np.sqrt(np.abs(variance_unbiased))
        z = confidence_interval * np.sqrt(X.shape[0]) / np.std(y)
        confidence = np.vectorize(st.norm.cdf)(z)
        confidence = np.mean(confidence, 1) * 100

        return confidence

    def train_final(self, tokens):
        new_X = self.remove_stops(tokens)
        all_counts = self.count_vec.transform(new_X)
        ext_X = self.tfid.transform(all_counts)
        output = self.model.predict(ext_X)
        confidence = self.calc_confidence(ext_X, output)
        decoded_output = self.embedding_decode(output, self.rnc)
        full_out = []
        for i in decoded_output:
            temp = []
            for j in np.where(i)[0]:
                temp.append(self.classes[j])
            full_out.append(temp)
        return full_out, confidence

    def parse_reports(self,reports):
        labels, items = [], []
        for r in reports:
            for s in r.sentences:
                for m in s.matches:
                    new_labels, new_items = self.parse_search([m.search])
                    labels.extend(new_labels)
                    items.extend(new_items)
        return labels, items

    @staticmethod
    def remove_stops(X):
        nlp = spacy.load('en_core_web_sm')
        new_X = []
        for sent in nlp.pipe(X):
            temp = []
            for tok in sent:
                if not tok.is_stop:
                    temp.append(tok.text)
            new_X.append(' '.join(temp))
        return(new_X)

    @staticmethod
    def remove_nones(y):
        new_y = []
        for i in y:
            temp = []
            for j in i:
                if(j == None):
                    temp.append('NO_TECHNIQUE')
                else:
                    temp.append(j)
            new_y.append(temp)
        return new_y

    @staticmethod
    def parse_training_data(training_data):
        labels, items = [], []
        for t in training_data:
            labels.append(t.code)
            items.append(t.description)
        return labels, items

    @staticmethod
    def parse_search(search):
        labels, items = [], []
        for s in search:
            if s.code is None:
                labels.append(['NO_TECHNIQUE'])
            else:
                labels.append([s.code])
            items.append(s.description)
        return labels, items

    # Functions from old BaseObject class
    def match(self, criteria):
        if not criteria:
            return self
        criteria_matches = []
        for k, v in criteria.items():
            if type(v) is tuple:
                for val in v:
                    if self.__getattribute__(k) == val:
                        criteria_matches.append(True)
            else:
                if self.__getattribute__(k) == v:
                    criteria_matches.append(True)
        if len(criteria_matches) == len(criteria) and all(criteria_matches):
            return self

    def update(self, field, value):
        if (value or type(value) == list) and (value != self.__getattribute__(field)):
            self.__setattr__(field, value)

    # @classmethod
    # def load(cls, dict_obj):
    #     if cls.load_schema:
    #         return cls.load_schema.load(dict_obj)
    #     elif cls.schema:
    #         return cls.schema.load(dict_obj)
    #     else:
    #         raise NotImplementedError
    # Functions from old BaseObject class

    # Functions from DataService class
    def data_svc_locate(self, object_name, match=None):
        result = [obj for obj in self.ram[object_name] if obj.match(match)]
        return result
    # End functions from old DataService class

    def create_report(self, filepath):
        report = Report(file=filepath, file_date=None)
        blob, tokens = report.generate_text_blob()
        for regex in self.regex_expressions:
            self.log.debug('[%s] Collecting %s indicator' % (report.id, regex['name']))
            self.regex_parser.find(regex, report, blob)
        model_arr = self.data_svc_locate('model', dict(name='base model'))
        inference = model_arr[0]
        full_out, confidence = inference.train_final(tokens)
        self.add_matches(full_out, confidence, report, tokens)
        return report

    def retrain_model(self, model_name):
        model = self.data_svc.locate('model', dict(name=model_name))
        asyncio.get_event_loop().create_task(self.get_service('machine_svc').retrain(model))

    def export_report(self, report_id, type):
        reports = self.data_svc.locate('reports', dict(id=report_id))
        return reports[0].export(type)

    def pull_rss_feed(self, rss_url):
        feed = feedparser.parse(rss_url['url'])
        for entry in feed.entries:
            url = entry.link
            new_report = Report(url=url)
            self.create_report(new_report)

    def pull_attack_refs(self):
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
    # End functions from tram_svc


class Search(object):
    def __init__(self, tag, name=None, description=None, code=None):
        self.id = '%s-%s-%s' % (name, code, description)
        self.tag = tag
        self.description = description
        self.name = name
        self.code = code

    def match(self, criteria):
        if not criteria:
            return self
        criteria_matches = []
        for k, v in criteria.items():
            if type(v) is tuple:
                for val in v:
                    if self.__getattribute__(k) == val:
                        criteria_matches.append(True)
            else:
                if self.__getattribute__(k) == v:
                    criteria_matches.append(True)
        if len(criteria_matches) == len(criteria) and all(criteria_matches):
            return self

    def store(self, ram):
        existing = retrieve(ram['search'], self.id)
        if not existing:
            ram['search'].append(self)
            return retrieve(ram['search'], self.id)
        return existing


class Sentence(object):
    def __init__(self, id=None, text=None):
        self.id = id if id else str(uuid.uuid4())
        self.text = text
        self.matches = []


class Status(Enum):
    QUEUE = 1
    TODO = 2
    REVIEW = 3
    COMPLETED = 4


class Match(object):
    def __init__(self, model=None, search=None, confidence=0, accepted=True, sentence=None, manual=False):
        self.id = str(uuid.uuid4())
        self.model = model
        self.search = search
        self.confidence = confidence
        self.accepted = accepted
        self.sentence = sentence
        self.manual = manual


class Report(object):

    MINIMUM_SENTENCE_LENGTH = 4
    EXPORTS = ['default', 'stix']

    @property
    def stage(self):
        return self.status

    def __init__(self, id=None, name=None, url=None, file=None, file_date=None, user=None, status=Status.TODO):
        if type(status) == str:
            status = Status[status]
        elif type(status) == int:
            status = Status(status)

        self.id = id if id else str(uuid.uuid4())
        self.status = status
        self.name = name
        self.url = url
        self.file = file
        self.file_date = file_date
        self.sentences = []
        self.matches = []
        self.completed_models = 0
        self.assigned_user = user

    def store(self, ram):
        existing = retrieve(ram['reports'], self.id)
        if not existing:
            ram['reports'].append(self)
            return retrieve(ram['reports'], self.id)
        existing.update('name', self.name)
        existing.update('assigned_user',self.assigned_user)
        existing.update('status', self.status)
        return existing


    def generate_text_blob(self):
        if self.url:
            content = self._get_soup(self.url)
            return str(content.text), self._clean_url_text(content)
        if self.file:
            content = FileParser.parse_file(self.file)
            return content, self._clean_file_text(content)

    def export(self, type): # what to return for each export type
        if type == 'stix':
            data = self.display
            output_stix = {}
            output_stix['objects'] = []
            output_stix['id'] = 'bundle--'+data['id']
            for sent in data['sentences']:
                if(len(sent['matches']) > 0):
                    for i in sent['matches']:
                        temp = {}
                        key = 'attack-pattern--'+i['id']
                        temp['type'] = 'attack-pattern'
                        temp['id'] = key
                        temp['name'] = i['search']['name']
                        temp['description'] = sent['text']
                        output_stix['objects'].append(temp)
            for ioc in data['matches']:
                temp = {}
                key = 'indicator--'+ioc['id']
                temp['id'] = key
                temp['indicator_type'] = ioc['search']['code']
                temp['name'] = ioc['search']['description']
                output_stix['objects'].append(temp)
                
            output_stix['type'] = 'bundle'
            output_stix['output_type'] = 'json'
            return output_stix
        else:
            output = self.display
            output['output_type'] = 'json'
            return output

    """ PRIVATE """

    @staticmethod
    def _get_soup(url):
        r = requests.get(url, verify=False)
        soup = BeautifulSoup(r.content, 'html.parser')
        for tag in ['script', 'style', 'button', 'nav', 'head', 'footer', 'a']:
            [s.extract() for s in soup(tag)]
        return soup

    @staticmethod
    def _clean_file_text(text):
        return nltk.sent_tokenize(text)

    def _clean_url_text(self, soup):
        text_list = []
        for text in soup.text.split('\n'):
            text = text.strip()
            for sentence in nltk.tokenize.sent_tokenize(text):
                if len(nltk.tokenize.word_tokenize(sentence)) > self.MINIMUM_SENTENCE_LENGTH:
                    text_list.append(sentence)
        return text_list
