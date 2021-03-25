from abc import ABC, abstractmethod
from io import BytesIO
import pathlib
import pickle
import time

from bs4 import BeautifulSoup
from django.db import transaction
import docx
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import pdfplumber
import nltk
import numpy as np

from typing import List

# The word model is overloaded in this scope, so a prefix is necessary
from tram import models as db_models

from sklearn.base import TransformerMixin
from sentence_transformers import SentenceTransformer


class SentenceEmbeddingTransformer(TransformerMixin):
    """Use pretrained sentence-transformer architectures as features to downstream classification models.

    """
    def __init__(self, model_name):
        self._model = SentenceTransformer(model_name)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        return self._model.encode(X)


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
            self.model = TramModel('stsb-roberta-large')
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
        sentences = []
        accepted_sentences = db_models.Sentence.objects.filter(disposition='accept')

        for accepted_sentence in accepted_sentences:
            mappings = db_models.Mapping.objects.filter(sentence=accepted_sentence)
            m = [Mapping(mapping.confidence, mapping.attack_technique.attack_id) for mapping in mappings]
            sentence = Sentence(accepted_sentence.text, accepted_sentence.order, m)
            sentences.append(sentence)

        return sentences

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
        return []

    def get_mappings(self, sentence):
        return []


class TramModel(Model):
    def __init__(self, model_name: str, max_iter: int=10_000):
        self.techniques_model = Pipeline([
            ("features", SentenceEmbeddingTransformer(model_name)),
            ("clf", OneVsRestClassifier(LogisticRegression(max_iter=max_iter)))
        ])

        self._t2i = {
            t.attack_id: i for i, t in enumerate(db_models.AttackTechnique.objects.all().order_by('attack_id'))
        }
        self._i2t = {v: k for k, v in self._t2i.items()}

    def __vectorize(self, sents: List[Sentence]):
        X = []
        y = []
        for sent in sents:
            X.append(sent.text)
            y_ = np.zeros(shape=(len(self._t2i)))
            for mapping in sent.mappings:
                y_[self._t2i[mapping.attack_technique]] = 1
            y.append(y_)
        return X, y

    def train(self):
        """
        Trains the model based on:
          1. ATT&CK data from disk
          2. User-annotated reports from models.Report
        """
        accepted_sents = self.get_training_data()
        X, y = self.__vectorize(accepted_sents)
        self.techniques_model.fit(X, y)

    def test(self):
        """Returns the f1 score
        """
        raise NotImplementedError()

    def get_indicators(self, text):
        return []

    def get_mappings(self, sentence):
        mappings = []
        # [0] because we're doing this one sentence at a time, but expected is batch in, batch out
        technique_preds = self.techniques_model.predict_proba([sentence])[0].tolist()
        for i, confidence in enumerate(technique_preds):
            if confidence >= 0.5:
                attack_technique = self._i2t[i]
                mapping = Mapping(confidence, attack_technique)
                mappings.append(mapping)
        return mappings
