from abc import ABC, abstractmethod
from datetime import datetime, timezone
from io import BytesIO
from os import path
import pathlib
import pickle
import time
import traceback

from bs4 import BeautifulSoup
from constance import config
from django.db import transaction
from django.conf import settings
import docx
import nltk
import pdfplumber
import re
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# The word model is overloaded in this scope, so a prefix is necessary
from tram import models as db_models


FROM_REPORT = 'from_report'
FROM_SENTENCE = 'from_sentence'
TO_TECHNIQUE = 'to_technique'
TO_GROUP = 'to_group'


class Sentence(object):
    def __init__(self, text, order, mappings):
        self.text = text
        self.order = order
        self.mappings = mappings


class TechniqueMapping(object):
    def __init__(self, confidence=0.0, attack_technique=None):
        self.confidence = confidence
        self.attack_technique = attack_technique

    def __repr__(self):
        return f'Confidence={self.confidence}; Technique={self.attack_technique}'


class GroupMapping(object):
    def __init__(self, confidence=0.0, attack_group=None):
        self.confidence = confidence
        self.attack_group = attack_group

    def __repr__(self):
        return f'Confidence={self.confidence}; Group={self.attack_group}'


class Report(object):
    def __init__(self, name, text, sentences):
        self.name = name
        self.text = text
        self.sentences = sentences  # Sentence objects


class TramModel(ABC):

    MAPPING_FROM = None
    MAPPING_TO = None

    def __init__(self):
        if self.MAPPING_FROM not in (FROM_REPORT, FROM_SENTENCE):
            raise ValueError('Invalid self.MAPPING_FROM. Must be one of FROM_REPORT or FROM_SENTENCE')

        if self.MAPPING_TO not in (TO_TECHNIQUE, TO_GROUP):
            raise ValueError('Invalid self.MAPPING_FROM. Must be one of TO_TECHNIQUE or TO_GROUP')

        self.sklearn_pipeline = self.get_sklearn_pipeline()
        self.mapping_class = TramModel._get_mapping_class(self.MAPPING_TO)
        self.mapping_database_model = TramModel._get_mapping_database_model(self.MAPPING_FROM, self.MAPPING_TO)

        self.last_trained = None
        self.average_f1_score = None
        self.detailed_f1_score = None

        if not isinstance(self.sklearn_pipeline, Pipeline):
            raise TypeError('get_model() must return an sklearn.pipeline.Pipeline instance')

    @staticmethod
    def _get_mapping_inputs(report, mapping_from):
        """
        TODO: Should the name stay?
        TODO: Sentence and Report probably need to have the same superclass or implement the same interface
        """
        if mapping_from == FROM_SENTENCE:
            return db_models.Sentence.objects.filter(report=report)
        elif mapping_from == FROM_REPORT:
            return [report, ]
        else:
            raise ValueError(f'Invalid mapping_from: {mapping_from}')

    @staticmethod
    def _get_mapping_class(mapping_to):
        if mapping_to == TO_TECHNIQUE:
            return TechniqueMapping
        elif mapping_to == TO_GROUP:
            return GroupMapping
        else:
            raise ValueError(f'Invalid mapping_to: {mapping_to}')

    @staticmethod
    def _get_mapping_database_model(mapping_from, mapping_to):
        if mapping_from == FROM_SENTENCE and mapping_to == TO_TECHNIQUE:
            return db_models.SentenceTechniqueMapping
        elif mapping_from == FROM_SENTENCE and mapping_to == TO_GROUP:
            return db_models.SentenceGroupMapping
        elif mapping_from == FROM_REPORT and mapping_to == TO_TECHNIQUE:
            return db_models.ReportTechniqueMapping
        elif mapping_from == FROM_REPORT and mapping_to == TO_GROUP:
            return db_models.ReportGroupMapping
        else:
            raise ValueError(f'Invalid mapping combination: from {mapping_from} to {mapping_to}')

    @abstractmethod
    def get_sklearn_pipeline(self):
        """Returns an sklearn.Pipeline that has fit() and predict() methods
        """

    def lemmatize(self, sentence):
        """
        Preprocess text by
        1) Lemmatizing - reducing words to their root, as a way to eliminate noise in the text
        2) Removing digits
        """
        lemma = nltk.stem.WordNetLemmatizer()

        # Lemmatize each word in sentence
        lemmatized_sentence = ' '.join([lemma.lemmatize(w) for w in sentence.rstrip().split()])
        lemmatized_sentence = re.sub(r'\d+', '', lemmatized_sentence)  # Remove digits with regex

        return lemmatized_sentence

    def get_training_data(self):
        """
        returns a tuple of lists, X, y.
        X is a list of lemmatized sentences; y is a list of Attack Techniques
        """
        X = []
        y = []

        mappings = self.mapping_database_model.get_accepted_mappings()
        for mapping in mappings:
            lemmatized_text = self.lemmatize(mapping.sentence.text)
            X.append(lemmatized_text)
            y.append(mapping.attack_technique.attack_id)

        return X, y

    def train(self):
        """
        Load and preprocess data. Train model pipeline
        """
        X, y = self.get_training_data()

        self.sklearn_pipeline.fit(X, y)  # Train classification model
        self.last_trained = datetime.now(timezone.utc)

    def test(self):
        """
        Return classification metrics based on train/test evaluation of the data
        Note: potential extension is to use cross-validation rather than a single train/test split
        """
        X, y = self.get_training_data()

        # Create training set and test set
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0, stratify=y)

        # Train model
        test_model = self.get_sklearn_pipeline()
        test_model.fit(X_train, y_train)

        # Generate predictions on test set
        y_predicted = test_model.predict(X_test)

        # TODO: Does this put labels and scores in the correct order?
        # Calculate an f1 score for each technique
        labels = sorted(list(set(y)))
        scores = f1_score(y_test, y_predicted, labels=list(set(y)), average=None)
        self.detailed_f1_score = sorted(zip(labels, scores), key=lambda t: t[1], reverse=True)

        # Average F1 score across techniques, weighted by the # of training examples per technique
        weighted_f1 = f1_score(y_test, y_predicted, average='weighted')
        self.average_f1_score = weighted_f1

    def process_report(self, report):
        # Input can be a report or a sentence
        mapping_inputs = TramModel._get_mapping_inputs(report, mapping_from=self.MAPPING_FROM)
        for mapping_input in mapping_inputs:
            mappings = self.get_mappings(mapping_input.text)
            for mapping in mappings:
                if not isinstance(mapping, self.mapping_class):
                    raise TypeError(f'Mapping is not of type {self.mapping_class}')

            db_mapping = self.mapping_database_model.from_mapping(mapping_input, mapping)
            db_mapping.save()

    def get_mappings(self, text):
        """
        Use trained model to predict the technique for a given sentence.
        """
        mappings = []

        techniques = self.sklearn_pipeline.classes_
        probs = self.sklearn_pipeline.predict_proba([text])[0]  # Probability is a range between 0-1

        # Create a list of tuples of (confidence, technique)
        confidences_and_techniques = zip(probs, techniques)
        for confidence_and_technique in confidences_and_techniques:
            confidence = confidence_and_technique[0] * 100
            attack_id = confidence_and_technique[1]
            if confidence < config.ML_CONFIDENCE_THRESHOLD:
                # Ignore proposed mappings below the confidence threshold
                continue
            mapping = self.mapping_class(confidence, attack_id)
            mappings.append(mapping)

        return mappings

    def save_to_file(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filepath):
        with open(filepath, 'rb') as f:
            model = pickle.load(f)  # nosec - Accept the risk until a better design is implemented

        assert cls == model.__class__
        return model


class ReportGroupModel(TramModel, ABC):
    MAPPING_FROM = FROM_REPORT
    MAPPING_TO = TO_GROUP


class ReportTechniqueModel(TramModel, ABC):
    MAPPING_FROM = FROM_REPORT
    MAPPING_TO = TO_TECHNIQUE


class SentenceGroupModel(TramModel, ABC):
    MAPPING_FROM = FROM_SENTENCE
    MAPPING_TO = TO_GROUP


class SentenceTechniqueModel(TramModel, ABC):
    MAPPING_FROM = FROM_SENTENCE
    MAPPING_TO = TO_TECHNIQUE


class DummySentenceTechniqueModel(SentenceTechniqueModel):
    def get_sklearn_pipeline(self):
        return Pipeline([
            ("features", CountVectorizer(lowercase=True, stop_words='english', min_df=3)),
            ("clf", DummyClassifier(strategy='uniform'))
        ])


class DummySentenceGroupModel(SentenceGroupModel):
    def get_sklearn_pipeline(self):
        return Pipeline([
            ("features", CountVectorizer(lowercase=True, stop_words='english', min_df=3)),
            ("clf", DummyClassifier(strategy='uniform'))
        ])


class DummyFulltextTechniqueModel(ReportTechniqueModel):
    def get_sklearn_pipeline(self):
        return Pipeline([
            ("features", CountVectorizer(lowercase=True, stop_words='english', min_df=3)),
            ("clf", DummyClassifier(strategy='uniform'))
        ])


class DummyFulltextGroupModel(ReportGroupModel):
    def get_sklearn_pipeline(self):
        return Pipeline([
            ("features", CountVectorizer(lowercase=True, stop_words='english', min_df=3)),
            ("clf", DummyClassifier(strategy='uniform'))
        ])


class NaiveBayesModel(SentenceTechniqueModel):
    def get_sklearn_pipeline(self):
        """
        Modeling pipeline:
        1) Features = document-term matrix, with stop words removed from the term vocabulary.
        2) Classifier (clf) = multinomial Naive Bayes
        """
        return Pipeline([
            ("features", CountVectorizer(lowercase=True, stop_words='english', min_df=3)),
            ("clf", MultinomialNB())
        ])


class LogisticRegressionModel(SentenceTechniqueModel):
    def get_sklearn_pipeline(self):
        """
        Modeling pipeline:
        1) Features = document-term matrix, with stop words removed from the term vocabulary.
        2) Classifier (clf) = multinomial logistic regression
        """
        return Pipeline([
            ("features", CountVectorizer(lowercase=True, stop_words='english', min_df=3)),
            ("clf", LogisticRegression())
        ])


class ModelManager(object):
    model_registry = {  # TODO: Add a hook to register user-created models
        'dummy-st': DummySentenceTechniqueModel,
        'dummy-sg': DummySentenceGroupModel,
        'dummy-ft': DummyFulltextTechniqueModel,
        'dummy-fg': DummyFulltextGroupModel,
        'nb': NaiveBayesModel,
        'logreg': LogisticRegressionModel,
    }

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

    def _sentence_tokenize(self, text):
        return nltk.sent_tokenize(text)

    def create_report(self, document):
        """
        Takes a document and performs the following:
            1. Creates a report in the database from the document
            2. Extracts sentences from the document and creates them in the database
            3. DOES NOT perform any mappings or machine learning

        Returns the saved Report object.
        """
        name = 'Report for %s' % pathlib.Path(document.docfile.path).name
        text = self._extract_text(document)
        report = db_models.Report(name=name, document=document, text=text)
        report.save()

        sentences = self._sentence_tokenize(text)
        order = 0
        for sentence in sentences:
            s = db_models.Sentence(text=sentence, document=document, order=order,
                                   report=report, disposition=None)
            s.save()

        return report

    @staticmethod
    def _extract_text(document):
        suffix = pathlib.Path(document.docfile.path).suffix
        if suffix == '.pdf':
            text = ModelManager._extract_pdf_text(document)
        elif suffix == '.docx':
            text = ModelManager._extract_docx_text(document)
        elif suffix == '.html':
            text = ModelManager._extract_html_text(document)
        else:
            raise ValueError('Unknown file suffix: %s' % suffix)

        return text

    @staticmethod
    def _extract_pdf_text(document):
        with pdfplumber.open(BytesIO(document.docfile.read())) as pdf:
            text = ''.join(page.extract_text() for page in pdf.pages if page.extract_text())

        if len(text) == 0:
            raise EOFError('No text was extracted from PDF')

        return text

    @staticmethod
    def _extract_html_text(document):
        html = document.docfile.read()
        soup = BeautifulSoup(html, features="html.parser")
        text = soup.get_text()
        return text

    @staticmethod
    def _extract_docx_text(document):
        parsed_docx = docx.Document(BytesIO(document.docfile.read()))
        text = ' '.join([paragraph.text for paragraph in parsed_docx.paragraphs])
        return text

    @staticmethod
    def get_model_instance(model_name):
        model_class = ModelManager.model_registry.get(model_name)
        if not model_class:
            raise ValueError('Unrecognized model: %s' % model_name)

        model_filepath = ModelManager.get_model_filepath(model_class)
        if path.exists(model_filepath):
            model_instance = model_class.load_from_file(model_filepath)
            print('%s loaded from %s' % (model_class.__name__, model_filepath))
        else:
            model_instance = model_class()
            print('%s loaded from __init__' % model_class.__name__)

        return model_instance

    def run_pipeline(self, model_names, run_forever=False):
        if len(model_names) == 0:
            raise ValueError('At least one ml_model must be specified')

        if len(model_names) > 10:
            raise ValueError('At most 10 ml_models can be specified')

        ml_models = [self.get_model_instance(model_name) for model_name in model_names]

        while True:
            jobs = db_models.DocumentProcessingJob.objects.filter(status='queued').order_by('created_on')
            for job in jobs:
                try:
                    with transaction.atomic():
                        filename = job.document.docfile.name
                        print('Processing Job #%d: %s' % (job.id, filename))
                        report = self.create_report(job.document)
                        print('Created report %s' % report.name)
                        import pdb
                        pdb.set_trace()
                        for ml_model in ml_models:
                            ml_model.process_report(report)
                        job.delete()
                except EOFError as e:
                    job.status = 'error'
                    job.message = str(e)
                    job.save()
                    print(f'Failed to create report for {filename}.')
                    print(traceback.format_exc())

            if not run_forever:
                return
            time.sleep(1)

    @staticmethod
    def get_model_filepath(model_class):
        filepath = settings.ML_MODEL_DIR + '/' + model_class.__name__ + '.pkl'
        return filepath

    @staticmethod
    def train_model(model_key):
        model = ModelManager.get_model_instance(model_key)
        model.train()
        model.test()
        filepath = ModelManager.get_model_filepath(model.__class__)
        model.save_to_file(filepath)
        print('Trained model saved to %s' % filepath)
        return

    @staticmethod
    def get_all_model_metadata():
        """
        Returns a list of model metadata for all models
        """
        all_model_metadata = []
        for model_key in ModelManager.model_registry.keys():
            model_metadata = ModelManager.get_model_metadata(model_key)
            all_model_metadata.append(model_metadata)

        all_model_metadata = sorted(all_model_metadata, key=lambda i: i['average_f1_score'], reverse=True)

        return all_model_metadata

    @staticmethod
    def get_model_metadata(model_key):
        """
        Returns a dict of model metadata for a particular ML model, identified by it's key
        """
        model = ModelManager.get_model_instance(model_key)
        model_name = model.__class__.__name__
        if model.last_trained is None:
            last_trained = 'Never trained'
            trained_techniques_count = 0
        else:
            last_trained = model.last_trained.strftime('%m/%d/%Y %H:%M:%S UTC')
            trained_techniques_count = len(model.detailed_f1_score)

        average_f1_score = round((model.average_f1_score or 0.0) * 100, 2)
        stored_scores = model.detailed_f1_score or []
        attack_ids = set([score[0] for score in stored_scores])
        attack_techniques = db_models.AttackTechnique.objects.filter(attack_id__in=attack_ids)
        detailed_f1_score = []
        for score in stored_scores:
            score_id = score[0]
            score_value = round(score[1] * 100, 2)

            attack_technique = attack_techniques.get(attack_id=score_id)
            detailed_f1_score.append({
                'technique': score_id,
                'technique_name': attack_technique.name,
                'attack_url': attack_technique.attack_url,
                'score': score_value
            })
        model_metadata = {
            'model_key': model_key,
            'name': model_name,
            'last_trained': last_trained,
            'trained_techniques_count': trained_techniques_count,
            'average_f1_score': average_f1_score,
            'detailed_f1_score': detailed_f1_score,
        }
        return model_metadata
