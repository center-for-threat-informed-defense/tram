from abc import ABC, abstractmethod
from io import BytesIO
import pathlib
import pickle
import random
import time

from bs4 import BeautifulSoup
from django.db import transaction
import docx
import pdfplumber
import nltk

# The word model is overloaded in this scope, so a prefix is necessary
from tram import models as db_models


class Indicator(object):
    def __init__(self, type_, value):
        self.type_ = type_
        self.value = value

    def __repr__(self):
        return 'Indicator: %s=%s' % (self.type_, self.value)


class Sentence(object):
    def __init__(self, text, order, mappings):
        self.text = text
        self.order = order
        self.mappings = mappings


class Mapping(object):
    def __init__(self, confidence=0.0, attack_technique=None):
        self.confidence = confidence
        self.attack_technique = attack_technique

    def __repr__(self):
        return 'Confidence=%f; Technique=%s' % (self.confidence, self.attack_technique)


class Report(object):
    def __init__(self, name, text, sentences, indicators=None):
        self.name = name
        self.text = text
        self.sentences = sentences  # Sentence objects
        self.indicators = indicators or []


class ModelManager(object):
    def __init__(self, model):
        if model == 'tram':
            self.model = TramModel()
        elif model == 'dummy':
            self.model = DummyModel()
        else:
            raise ValueError('Unkown model: %s' % model)

    def _save_report(self, report, document):
        rpt = db_models.Report(
            name=report.name,
            document=document,
            text=report.text,
            ml_model=self.model.__class__.__name__
        )
        rpt.save()

        for indicator in report.indicators:
            ind = db_models.Indicator(
                report=rpt,
                indicator_type=indicator.type_,
                value=indicator.value
            )
            ind.save()

        for sentence in report.sentences:
            s = db_models.Sentence(
                text=sentence.text,
                order=sentence.order,
                document=document,
                report=rpt,
                disposition=None,
            )
            s.save()

            for mapping in sentence.mappings:
                if mapping.attack_technique:
                    technique = db_models.AttackTechnique.objects.get(attack_id=mapping.attack_technique)
                else:
                    technique = None

                m = db_models.Mapping(
                    report=rpt,
                    sentence=s,
                    attack_technique=technique,
                    confidence=mapping.confidence,
                )
                m.save()

    def run_model(self):
        while True:
            jobs = db_models.DocumentProcessingJob.objects.all().order_by('created_on')
            for job in jobs:
                print('Processing Job #%d: %s' % (job.id, job.document.docfile.name))
                report = self.model.process_job(job)
                with transaction.atomic():
                    self._save_report(report, job.document)
                    job.delete()
                print('Created report %s' % report.name)
            time.sleep(1)

    def train_model(self):
        return self.model.train()

    def test_model(self):
        return self.model.test()


class Model(ABC):
    def __init__(self):
        self._technique_ids = None

    @abstractmethod
    def train(self):
        """Trains the model based on:
           1. Source data (??)
           2. Reports in the database

        Returns ???
        """

    @abstractmethod
    def test(self):
        """Returns the f1 score as a float
        """

    @property
    def technique_ids(self):
        if not self._technique_ids:
            self._technique_ids = self.get_attack_technique_ids()
        return self._technique_ids

    def _get_report_name(self, job):
        name = pathlib.Path(job.document.docfile.path).name
        return 'Report for %s' % name

    def _extract_text(self, document):
        suffix = pathlib.Path(document.docfile.path).suffix
        if suffix == '.pdf':
            text = self._extract_pdf_text(document)
        elif suffix == '.docx':
            text = self._extract_docx_text(document)
        elif suffix == '.html':
            text = self._extract_html_text(document)
        else:
            raise ValueError('Unknown file suffix: %s' % suffix)

        return text

    def get_training_data(self):
        """Returns a list of base.Sentence objects where there is an accepted mapping"""
        # TODO: Get a list of accepted sentences and their mappings
        pass

    def get_attack_technique_ids(self):
        techniques = [t.attack_id for t in db_models.AttackTechnique.objects.all().order_by('attack_id')]
        if len(techniques) == 0:
            raise ValueError('Zero techniques found. Maybe run `python manage.py attackdata load` ?')
        return techniques

    @abstractmethod
    def get_indicators(self, text):
        """
        Returns an array of indicator objects
        """

    @abstractmethod
    def get_mappings(self, sentence):
        """
        Returns a list of Mapping objects for the sentence.
        Returns an empty list if there are no Mappings
        """

    def _sentence_tokenize(self, text):
        return nltk.sent_tokenize(text)

    def _extract_pdf_text(self, document):
        with pdfplumber.open(BytesIO(document.docfile.read())) as pdf:
            text = ''.join(page.extract_text() for page in pdf.pages)

        return text

    def _extract_html_text(self, document):
        html = document.docfile.read()
        soup = BeautifulSoup(html, features="html.parser")
        text = soup.get_text()
        return text

    def _extract_docx_text(self, document):
        parsed_docx = docx.Document(BytesIO(document.docfile.read()))
        text = ' '.join([paragraph.text for paragraph in parsed_docx.paragraphs])
        return text

    def process_job(self, job):
        name = self._get_report_name(job)
        text = self._extract_text(job.document)
        sentences = self._sentence_tokenize(text)
        indicators = self.get_indicators(text)

        report_sentences = []
        order = 0
        for sentence in sentences:
            mappings = self.get_mappings(sentence)
            s = Sentence(text=sentence, order=order, mappings=mappings)
            order += 1
            report_sentences.append(s)
            # TODO: Implement
            # else:
            #    if list is empty, append None mapping with 100% confidence

        report = Report(name, text, report_sentences, indicators)
        return report

    def save_to_file(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        assert cls == model.__class__
        return model


class DummyModel(Model):
    def train(self):
        pass

    def test(self):
        pass

    def get_indicators(self, text):
        import uuid
        indicators = []
        for i in range(3):
            ind = Indicator(type_='MD5', value=uuid.uuid4().hex)
            indicators.append(ind)
        return indicators

    def _pick_random_techniques(self):
        """Returns a list of 0-4 randomly selected ATTACK Technique IDs"""
        num_techniques = random.randint(0, 4)
        techniques = random.choices(self.technique_ids, k=num_techniques)
        return techniques

    def get_mappings(self, sentence):
        mappings = []
        attack_techniques = self._pick_random_techniques()
        for attack_technique in attack_techniques:
            confidence = random.uniform(0.0, 100.0)
            mapping = Mapping(confidence, attack_technique)
            mappings.append(mapping)

        return mappings


class TramModel(Model):
    def train(self):
        """
        Trains the model based on:
          1. ATT&CK data from disk
          2. User-annotated reports from models.Report
        """
        raise NotImplementedError()

    def test(self):
        """Returns the f1 score
        """
        raise NotImplementedError()

    def get_indicators(self, text):
        raise NotImplementedError()

    def get_mappings(self, sentence):
        raise NotImplementedError()
