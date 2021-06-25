from abc import ABC, abstractmethod
from io import BytesIO
from os import path
import pathlib
import pickle
import random
import time

from bs4 import BeautifulSoup
from django.db import transaction
from django.conf import settings
import docx
import nltk
import numpy as np
import pandas as pd
import pdfplumber
import re
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from typing import List

# The word model is overloaded in this scope, so a prefix is necessary
from tram import models as db_models

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
    def __init__(self, name, text, sentences):
        self.name = name
        self.text = text
        self.sentences = sentences  # Sentence objects


class Model(ABC):
    def __init__(self):
        self._technique_ids = None

    def train(self):
        """
        Load and preprocess data. Train model pipeline
        """
        X, y = self._load_and_vectorize_data()
        X, y = self._filter_low_data_classes(X, y)
        X = self._preprocess_text(X)
        self.techniques_model.fit(X, y)  # Train classification model

    def test(self):
        """
        Return classification metrics based on train/test evaluation of the data
        Note: potential extension is to use cross-validation rather than a single train/test split
        """
        X, y = self._load_and_vectorize_data()
        X, y = self._filter_low_data_classes(X, y)
        X = self._preprocess_text(X)

        # Create training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            shuffle=True, random_state=0, stratify=y)

        # Train model
        self.techniques_model.fit(X_train, y_train)

        # Generate predictions on test set
        y_predicted = self.techniques_model.predict(X_test)

        """
        # Average training and test accuracy
        print('Training accuracy:', self.techniques_model.score(X_train, y_train))
        print('Test accuracy:', self.techniques_model.score(X_test, y_test))

        # Precision, recall, F1 score by technique
        print(classification_report(y_test, y_predicted, zero_division = 0))
        """

        # Average F1 score across techniques, weighted by the # of training examples per technique
        weighted_f1 = f1_score(y_test, y_predicted, average='weighted')
        return str(weighted_f1)  # TODO: Should this be a float?

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

    def _preprocess_text(self, sentences):
        """
        Preprocess text by
        1) Lemmatizing - reducing words to their root, as a way to eliminate noise in the text
        2) Removing digits
        """
        lemma = nltk.stem.WordNetLemmatizer()
        preprocessed_sentences = []
        for sentence in sentences:
            # Lemmatize each word in sentence
            sentence = ' '.join([lemma.lemmatize(w) for w in sentence.rstrip().split()])
            sentence = re.sub(r'\d+', '', sentence)  # Remove digits with regex
            preprocessed_sentences.append(sentence)
        return preprocessed_sentences

    def _load_and_vectorize_data(self):
        """
        Load training data from database.
        Store sentence text in vector X
        Store attack technique in vector y
        """
        X = []
        y = []
        accepted_sents = self.get_training_data()
        for sent_obj in accepted_sents:
            if sent_obj.mappings:  # Only store sentences with a labeled technique
                sentence = sent_obj.text
                technique_label = sent_obj.mappings[0].attack_technique
                technique = technique_label[0:5]  # Cut string to technique level. leave out sub-technique
                X.append(sentence)
                y.append(technique)
        return X, y

    def _filter_low_data_classes(self, X, y, n_examples_threshold=5):
        """
        Only retain data for classes with at least examples above n_examples_threshold
        Prevents predictions being made for classes without enough data to learn a
        generalizable pattern (i.e., too much noise)

        TODO: filter classes before data gets loaded
        """
        df = pd.DataFrame({'X': X, 'y': y})
        classes_to_keep = df['y'].value_counts().index[df['y'].value_counts() >= n_examples_threshold]
        df = df.loc[df['y'].isin(classes_to_keep)]
        X = df['X']
        y = df['y']
        return X, y

    def _inspect_estimated_parameters(self):
        """
        For Naive Bayes, we can obtain the log probability of a term given a technique, P(term|y)
        Here we print the top 5 terms by technique
        """
        classifier = self.techniques_model.named_steps['clf']
        vocabulary = self.techniques_model.named_steps['features'].get_feature_names()
        model_classes = self.techniques_model.classes_
        for technique_idx in range(len(model_classes)):
            technique = model_classes[technique_idx]
            print(technique)
            prob_sorted = classifier.feature_log_prob_[technique_idx, :].argsort()[::-1]
            print(np.take(vocabulary, prob_sorted[:5]))

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

        report_sentences = []
        order = 0
        for sentence in sentences:
            mappings = self.get_mappings(sentence)
            s = Sentence(text=sentence, order=order, mappings=mappings)
            order += 1
            report_sentences.append(s)

        report = Report(name, text, report_sentences)
        return report

    def save_to_file(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)  # nosec - Accept the risk until a better design is implemented

        assert cls == model.__class__
        return model


class DummyModel(Model):
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


class NaiveBayesModel(Model):
    def __init__(self):
        """
        Modeling pipeline:
        1) Features = document-term matrix, with stop words removed from the term vocabulary.
        2) Classifier (clf) = multinomial Naive Bayes
        """
        self.techniques_model = Pipeline([
            ("features", CountVectorizer(lowercase=True, stop_words='english', min_df=3)),
            ("clf", MultinomialNB())
        ])

    def get_mappings(self, sentence):
        """
        Use trained model to predict the technique for a given sentence.
        """
        mappings = []

        # Output top 3 techniques based on prediction confidence
        techniques = self.techniques_model.classes_
        probs = self.techniques_model.predict_proba([sentence])[0]
        top_3_conf_techniques = sorted(zip(probs, techniques), reverse=True)[:3]
        for res in top_3_conf_techniques:
            conf = res[0] * 100
            attack_technique = res[1]
            mapping = Mapping(conf, attack_technique)
            mappings.append(mapping)

        return mappings


class LogisticRegressionModel(Model):
    def __init__(self):
        """
        Modeling pipeline:
        1) Features = document-term matrix, with stop words removed from the term vocabulary.
        2) Classifier (clf) = multinomial logistic regression
        """
        self.techniques_model = Pipeline([
            ("features", CountVectorizer(lowercase=True, stop_words='english', min_df=3)),
            ("clf", LogisticRegression())
        ])

    def get_mappings(self, sentence):
        """
        Use trained model to predict the technique for a given sentence.
        """
        mappings = []

        # Output top 3 techniques based on prediction confidence
        techniques = self.techniques_model.classes_
        probs = self.techniques_model.predict_proba([sentence])[0]
        top_3_conf_techniques = sorted(zip(probs, techniques), reverse=True)[:3]
        for res in top_3_conf_techniques:
            conf = res[0] * 100
            attack_technique = res[1]
            mapping = Mapping(conf, attack_technique)
            mappings.append(mapping)

        return mappings


class TramModel(Model):
    def __init__(self, model_name: str = 'stsb-roberta-large', max_iter: int = 10_000):
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


class ModelManager(object):
    model_registry = {  # TODO: Add a hook to register user-created models
        'tram': TramModel,
        'dummy': DummyModel,
        'nb': NaiveBayesModel,
        'logreg': LogisticRegressionModel,

    }

    def __init__(self, model):
        model_class = self.model_registry.get(model)
        if not model_class:
            raise ValueError('Unrecognized model: %s' % model)

        model_filepath = self.get_model_filepath(model_class)
        if path.exists(model_filepath):
            self.model = model_class.load_from_file(model_filepath)
            print('%s loaded from %s' % (model_class.__name__, model_filepath))
        else:
            self.model = model_class()
            print('%s loaded from __init__' % model_class.__name__)

    def _save_report(self, report, document):
        rpt = db_models.Report(
            name=report.name,
            document=document,
            text=report.text,
            ml_model=self.model.__class__.__name__
        )
        rpt.save()

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

    def get_model_filepath(self, model_class):
        filepath = settings.ML_MODEL_DIR + '/' + model_class.__name__ + '.pkl'
        return filepath

    def train_model(self):
        self.model.train()
        filepath = self.get_model_filepath(self.model.__class__)
        self.model.save_to_file(filepath)
        print('Trained model saved to %s' % filepath)
        return

    def test_model(self):
        return self.model.test()
