import logging
import pathlib
import pickle
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from io import BytesIO
from os import path

import docx
import nltk
import pdfplumber
from bs4 import BeautifulSoup
from constance import config
from django.conf import settings
from django.db import transaction
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# The word model is overloaded in this scope, so a prefix is necessary
from tram import models as db_models


logger = logging.getLogger(__name__)


class Sentence(object):
    def __init__(self, text, order, mappings):
        self.text = text
        self.order = order
        self.mappings = mappings


class Mapping(object):
    def __init__(self, confidence=0.0, attack_id=None):
        self.confidence = confidence
        self.attack_id = attack_id

    def __repr__(self):
        return "Confidence=%f; Attack ID=%s" % (self.confidence, self.attack_id)


class Report(object):
    def __init__(self, name, text, sentences):
        self.name = name
        self.text = text
        self.sentences = sentences  # Sentence objects


class ExtractionError(Exception):
    """An error that happens while extracting text from a source."""


class SKLearnModel(ABC):
    """
    TODO:
    1. Move text extraction and tokenization out of the SKLearnModel
    """

    def __init__(self):
        self.techniques_model = self.get_model()
        self.last_trained = None
        self.average_f1_score = None
        self.detailed_f1_score = None

        if not isinstance(self.techniques_model, Pipeline):
            raise TypeError(
                "get_model() must return an sklearn.pipeline.Pipeline instance"
            )

    @abstractmethod
    def get_model(self):
        """Returns an sklearn.Pipeline that has fit() and predict() methods"""

    def train(self):
        """
        Load and preprocess data. Train model pipeline
        """
        X, y = self.get_training_data()

        self.techniques_model.fit(X, y)  # Train classification model
        self.last_trained = datetime.now(timezone.utc)

    def test(self):
        """
        Return classification metrics based on train/test evaluation of the data
        Note: potential extension is to use cross-validation rather than a single train/test split
        """
        X, y = self.get_training_data()

        # Create training set and test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=0, stratify=y
        )

        # Train model
        test_model = self.get_model()
        test_model.fit(X_train, y_train)

        # Generate predictions on test set
        y_predicted = test_model.predict(X_test)

        # TODO: Does this put labels and scores in the correct order?
        # Calculate an f1 score for each technique
        labels = sorted(list(set(y)))
        scores = f1_score(y_test, y_predicted, labels=list(set(y)), average=None)
        self.detailed_f1_score = sorted(
            zip(labels, scores), key=lambda t: t[1], reverse=True
        )

        # Average F1 score across techniques, weighted by the # of training examples per technique
        weighted_f1 = f1_score(y_test, y_predicted, average="weighted")
        self.average_f1_score = weighted_f1

    def _get_report_name(self, job):
        name = pathlib.Path(job.document.docfile.path).name
        return "Report for %s" % name

    def _extract_text(self, document):
        suffix = pathlib.Path(document.docfile.path).suffix
        if suffix == ".pdf":
            text = self._extract_pdf_text(document)
        elif suffix == ".docx":
            text = self._extract_docx_text(document)
        elif suffix == ".html":
            text = self._extract_html_text(document)
        elif suffix == ".txt":
            text = self._extract_plain_text(document)
        else:
            raise ValueError("Unknown file suffix: %s" % suffix)

        return text

    def lemmatize(self, sentence):
        """
        Preprocess text by
        1) Lemmatizing - reducing words to their root, as a way to eliminate noise in the text
        2) Removing digits
        """
        lemma = nltk.stem.WordNetLemmatizer()

        # Lemmatize each word in sentence
        lemmatized_sentence = " ".join(
            [lemma.lemmatize(w) for w in sentence.rstrip().split()]
        )
        lemmatized_sentence = re.sub(
            r"\d+", "", lemmatized_sentence
        )  # Remove digits with regex

        return lemmatized_sentence

    def get_training_data(self):
        """
        returns a tuple of lists, X, y.
        X is a list of lemmatized sentences; y is a list of Attack Techniques
        """
        X = []
        y = []
        mappings = db_models.Mapping.get_accepted_mappings()
        for mapping in mappings:
            lemmatized_sentence = self.lemmatize(mapping.sentence.text)
            X.append(lemmatized_sentence)
            y.append(mapping.attack_object.attack_id)

        return X, y

    def get_attack_object_ids(self):
        objects = [
            obj.attack_id
            for obj in db_models.AttackObject.objects.all().order_by("attack_id")
        ]
        if len(objects) == 0:
            raise ValueError(
                "Zero techniques found. Maybe run `python manage.py attackdata load` ?"
            )
        return objects

    def get_mappings(self, sentence):
        """
        Use trained model to predict the technique for a given sentence.
        """
        mappings = []

        techniques = self.techniques_model.classes_
        probs = self.techniques_model.predict_proba([sentence])[
            0
        ]  # Probability is a range between 0-1

        # Create a list of tuples of (confidence, technique)
        confidences_and_techniques = zip(probs, techniques)
        for confidence_and_technique in confidences_and_techniques:
            confidence = confidence_and_technique[0] * 100
            attack_technique = confidence_and_technique[1]
            if confidence < config.ML_CONFIDENCE_THRESHOLD:
                # Ignore proposed mappings below the confidence threshold
                continue
            mapping = Mapping(confidence, attack_technique)
            mappings.append(mapping)

        return mappings

    def _sentence_tokenize(self, text):
        return nltk.sent_tokenize(text)

    def _extract_pdf_text(self, document):
        with pdfplumber.open(BytesIO(document.docfile.read())) as pdf:
            text = "".join(page.extract_text() for page in pdf.pages)

        # If no text was extracted, then something went wrong.
        if not text:
            raise ExtractionError(
                "Could not extract text from PDF. Check that the"
                " PDF contains selectable text."
            )

        return text

    def _extract_html_text(self, document):
        html = document.docfile.read()
        soup = BeautifulSoup(html, features="html.parser")
        text = soup.get_text()
        return text

    def _extract_docx_text(self, document):
        parsed_docx = docx.Document(BytesIO(document.docfile.read()))
        text = " ".join([paragraph.text for paragraph in parsed_docx.paragraphs])
        return text

    def _extract_plain_text(self, document):
        text = document.docfile.read().decode("UTF-8")
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
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filepath):
        with open(filepath, "rb") as f:
            model = pickle.load(f)  # nosec
            # accept risk until better design implemented
        assert cls == model.__class__
        return model


class DummyModel(SKLearnModel):
    def get_model(self):
        return Pipeline(
            [
                (
                    "features",
                    CountVectorizer(lowercase=True, stop_words="english", min_df=3),
                ),
                ("clf", DummyClassifier(strategy="uniform")),
            ]
        )


class NaiveBayesModel(SKLearnModel):
    def get_model(self):
        """
        Modeling pipeline:
        1) Features = document-term matrix, with stop words removed from the term vocabulary.
        2) Classifier (clf) = multinomial Naive Bayes
        """
        return Pipeline(
            [
                (
                    "features",
                    CountVectorizer(lowercase=True, stop_words="english", min_df=3),
                ),
                ("clf", MultinomialNB()),
            ]
        )


class LogisticRegressionModel(SKLearnModel):
    def get_model(self):
        """
        Modeling pipeline:
        1) Features = document-term matrix, with stop words removed from the term vocabulary.
        2) Classifier (clf) = multinomial logistic regression
        """
        return Pipeline(
            [
                (
                    "features",
                    CountVectorizer(lowercase=True, stop_words="english", min_df=3),
                ),
                ("clf", LogisticRegression()),
            ]
        )


class MLPClassifierModel(SKLearnModel):
    def get_model(self):
        """
        Modeling pipeline:
        1) Features = document-term matrix, with stop words removed from the term vocabulary.
        2) Classifier (clf) = multi-layer perceptron classifier
        """
        return Pipeline(
            [
                (
                    "features",
                    CountVectorizer(lowercase=True, stop_words="english", min_df=3),
                ),
                ("clf", MLPClassifier(max_iter=1000)),
            ]
        )


class ModelManager(object):
    model_registry = {  # TODO: Add a hook to register user-created models
        "dummy": DummyModel,
        "nb": NaiveBayesModel,
        "logreg": LogisticRegressionModel,
        "nn_cls": MLPClassifierModel,
    }

    def __init__(self, model):
        model_class = self.model_registry.get(model)
        if not model_class:
            raise ValueError("Unrecognized model: %s" % model)

        model_filepath = self.get_model_filepath(model_class)
        if path.exists(model_filepath):
            self.model = model_class.load_from_file(model_filepath)
            logger.info("%s loaded from %s", model_class.__name__, model_filepath)
        else:
            self.model = model_class()
            logger.info("%s loaded from __init__", model_class.__name__)

    def _save_report(self, report, document):
        rpt = db_models.Report(
            name=report.name,
            document=document,
            text=report.text,
            ml_model=self.model.__class__.__name__,
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
                if mapping.attack_id:
                    obj = db_models.AttackObject.objects.get(
                        attack_id=mapping.attack_id
                    )
                else:
                    obj = None

                m = db_models.Mapping(
                    report=rpt,
                    sentence=s,
                    attack_object=obj,
                    confidence=mapping.confidence,
                )
                m.save()

    def run_model(self, run_forever=False):
        while True:
            jobs = db_models.DocumentProcessingJob.objects.filter(
                status="queued"
            ).order_by("created_on")
            for job in jobs:
                filename = job.document.docfile.name
                logger.info("Processing Job #%d: %s", job.id, filename)
                try:
                    report = self.model.process_job(job)
                    with transaction.atomic():
                        self._save_report(report, job.document)
                        job.delete()
                    logger.info("Created report %s", report.name)
                except Exception as ex:
                    job.status = "error"
                    job.message = str(ex)
                    job.save()
                    logger.exception("Failed to create report for %s.", filename)

            if not run_forever:
                return
            time.sleep(1)

    def get_model_filepath(self, model_class):
        filepath = settings.ML_MODEL_DIR + "/" + model_class.__name__ + ".pkl"
        return filepath

    def train_model(self):
        self.model.train()
        self.model.test()
        filepath = self.get_model_filepath(self.model.__class__)
        self.model.save_to_file(filepath)
        logger.info("Trained model saved to %s" % filepath)
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

        all_model_metadata = sorted(
            all_model_metadata, key=lambda i: i["average_f1_score"], reverse=True
        )

        return all_model_metadata

    @staticmethod
    def get_model_metadata(model_key):
        """
        Returns a dict of model metadata for a particular ML model, identified by it's key
        """
        mm = ModelManager(model_key)
        model_name = mm.model.__class__.__name__
        if mm.model.last_trained is None:
            last_trained = "Never trained"
            trained_techniques_count = 0
        else:
            last_trained = mm.model.last_trained.strftime("%m/%d/%Y %H:%M:%S UTC")
            trained_techniques_count = len(mm.model.detailed_f1_score)

        average_f1_score = round((mm.model.average_f1_score or 0.0) * 100, 2)
        stored_scores = mm.model.detailed_f1_score or []
        attack_ids = set([score[0] for score in stored_scores])
        attack_techniques = db_models.AttackObject.objects.filter(
            attack_id__in=attack_ids
        )
        detailed_f1_score = []
        for score in stored_scores:
            score_id = score[0]
            score_value = round(score[1] * 100, 2)

            attack_technique = attack_techniques.get(attack_id=score_id)
            detailed_f1_score.append(
                {
                    "technique": score_id,
                    "technique_name": attack_technique.name,
                    "attack_url": attack_technique.attack_url,
                    "score": score_value,
                }
            )
        model_metadata = {
            "model_key": model_key,
            "name": model_name,
            "last_trained": last_trained,
            "trained_techniques_count": trained_techniques_count,
            "average_f1_score": average_f1_score,
            "detailed_f1_score": detailed_f1_score,
        }
        return model_metadata
