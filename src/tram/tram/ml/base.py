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

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import chi2, SelectPercentile
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from nltk import word_tokenize
from sklearn.feature_extraction import text as tsk

# The word model is overloaded in this scope, so a prefix is necessary
from tram import models as db_models


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


class SKLearnModel(ABC):
    def __init__(self):
        self._technique_ids = None
        self.techniques_model = self.get_model()
        self.last_trained = None
        self.average_f1_score = None
        self.detailed_f1_score = None

        if not isinstance(self.techniques_model, Pipeline):
            raise TypeError('get_model() must return an sklearn.pipeline.Pipeline instance')

    @abstractmethod
    def get_model(self):
        """Returns an sklearn.Pipeline that has fit() and predict() methods
        """

    def train(self):
        """
        Load and preprocess data. Train model pipeline
        """
        X, y = self._load_and_vectorize_data()
        X = self._preprocess_text(X)
        self.techniques_model.fit(X, y)  # Train classification model
        self.last_trained = datetime.now(timezone.utc)

    def test(self):
        """
        Return classification metrics based on train/test evaluation of the data
        Note: potential extension is to use cross-validation rather than a single train/test split
        """
        X, y = self._load_and_vectorize_data()
        X = self._preprocess_text(X)

        # Check number of sentences per technique
        y_counts = {}
        for technique in y:
            if technique not in y_counts:
                y_counts[technique] = 0
            y_counts[technique] += 1

        # Create training set and test set
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0, stratify=y)

        # Train model
        test_model = self.get_model()
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
                # TODO: The below line omits all but the first mapped attack technique
                technique_label = sent_obj.mappings[0].attack_technique
                # TODO: The below line causes a reduction in the number of unique techniques, which will be
                #       surprising for an end user
                technique = technique_label[0:5]  # Cut string to technique level. leave out sub-technique
                X.append(sentence)
                y.append(technique)
        return X, y

    def get_training_data(self):
        """Returns a list of base.Sentence objects where there the number
           of accepted mappings is above the configured threshold (ML_ACCEPT_THRESHOLD)
        """
        # TODO: Refactor for readability and performance
        # Get Attack techniques that have >= the required amount of positive examples
        attack_techniques = db_models.AttackTechnique.get_sentence_counts(accept_threshold=config.ML_ACCEPT_THRESHOLD)
        # Get mappings for the attack techniques above threshold
        mappings = db_models.Mapping.objects.filter(attack_technique__in=attack_techniques)

        # For the mappings that are above threshold, identify the sentences
        training_sentence_ids = set()
        for mapping in mappings:
            training_sentence_ids.add(mapping.sentence.id)

        sentences = []
        for training_sentence_id in training_sentence_ids:
            training_sentence = db_models.Sentence.objects.get(id=training_sentence_id)
            mappings = db_models.Mapping.objects.filter(sentence=training_sentence)
            m = [Mapping(mapping.confidence, mapping.attack_technique.attack_id) for mapping in mappings]
            sentence = Sentence(training_sentence.text, training_sentence.order, m)
            sentences.append(sentence)

        return sentences

    def get_attack_technique_ids(self):
        techniques = [t.attack_id for t in db_models.AttackTechnique.objects.all().order_by('attack_id')]
        if len(techniques) == 0:
            raise ValueError('Zero techniques found. Maybe run `python manage.py attackdata load` ?')
        return techniques

    def get_mappings(self, sentence):
        """
        Use trained model to predict the technique for a given sentence.
        """
        mappings = []

        techniques = self.techniques_model.classes_
        probs = self.techniques_model.predict_proba([sentence])[0]  # Probability is a range between 0-1

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


class DummyModel(SKLearnModel):
    def get_model(self):
        return Pipeline([
            ("features", CountVectorizer(lowercase=True, stop_words='english', min_df=3)),
            ("clf", DummyClassifier(strategy='uniform'))
        ])


class NaiveBayesModel(SKLearnModel):
    def get_model(self):
        """
        Modeling pipeline:
        1) Features = document-term matrix, with stop words removed from the term vocabulary.
        2) Classifier (clf) = multinomial Naive Bayes
        """
        return Pipeline([
            ("features", CountVectorizer(lowercase=True, stop_words='english', min_df=3)),
            ("clf", MultinomialNB())
        ])


class LogisticRegressionModel(SKLearnModel):
    def get_model(self):
        """
        Modeling pipeline:
        1) Features = document-term matrix, with stop words removed from the term vocabulary.
        2) Classifier (clf) = multinomial logistic regression
        """
        return Pipeline([
            ("features", CountVectorizer(lowercase=True, stop_words='english', min_df=3)),
            ("clf", LogisticRegression())
        ])

class FullReportModel(SKLearnModel):
    
    class TextSelector(BaseEstimator, TransformerMixin):
        """
        Transformer to select a single column from the data frame to perform additional transformations on
        Use on text columns in the data
        """
        def __init__(self, key):
            self.key = key

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    
    def clean_text(text):
        """
        Cleaning up the words contractions, unusual spacing, non-word characters and any computer science
        related terms that hinder the classification.
        """
        text = str(text)
        text = text.lower()
        text = re.sub("\r\n", "\t", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"\'scuse", " excuse ", text)
        text = re.sub('(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.)\{3\}(?:25[0-5] |2[0-4][0-9]|[01]?[0-9][0-9]?)(/([0-2][0-9]|3[0-2]|[0-9]))?', 'IPv4', text)
        text = re.sub('\b(CVE\-[0-9]{4}\-[0-9]{4,6})\b', 'CVE', text)
        text = re.sub('\b([a-z][_a-z0-9-.]+@[a-z0-9-]+\.[a-z]+)\b', 'email', text)
        text = re.sub('\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', 'IP', text)
        text = re.sub('\b([a-f0-9]{32}|[A-F0-9]{32})\b', 'MD5', text)
        text = re.sub('\b((HKLM|HKCU)\\[\\A-Za-z0-9-_]+)\b', 'registry', text)
        text = re.sub('\b([a-f0-9]{40}|[A-F0-9]{40})\b', 'SHA1', text)
        text = re.sub('\b([a-f0-9]{64}|[A-F0-9]{64})\b', 'SHA250', text)
        text = re.sub('http(s)?:\\[0-9a-zA-Z_\.\-\\]+.', 'URL', text)
        text = re.sub('CVE-[0-9]{4}-[0-9]{4,6}', 'vulnerability', text)
        text = re.sub('[a-zA-Z]{1}:\\[0-9a-zA-Z_\.\-\\]+', 'file', text)
        text = re.sub('\b[a-fA-F\d]{32}\b|\b[a-fA-F\d]{40}\b|\b[a-fA-F\d]{64}\b', 'hash', text)
        text = re.sub('x[A-Fa-f0-9]{2}', ' ', text)
        text = re.sub('\W', ' ', text)
        text = re.sub('\s+', ' ', text)
        text = text.strip(' ')
        return text
    
    
    class StemTokenizer(object):
        def __init__(self):
            self.st = EnglishStemmer()
            
        def __call__(self, doc):
            return [self.st.stem(t) for t in word_tokenize(doc)]
    

    def _preprocess_text(self, sentences):
        lemma = nltk.stem.WordNetLemmatizer()
        preprocessed_sentences = ""
        for sentence in sentences:
            # Lemmatize each word in sentence
            sentence = ' '.join([lemma.lemmatize(w) for w in sentence.rstrip().split()])
            sentence = re.sub(r'\d+', '', sentence)  # Remove digits with regex
            preprocessed_sentences = preprocessed_sentences + ' ' + sentence
        return [preprocessed_sentences]

    def _sentence_tokenize(self, text):
        return [text]

    def get_model(self):
        stop_words = stopwords.words('english')
        new_stop_words = ["'ll", "'re", "'ve", 'ha', 'wa',"'d", "'s", 'abov', 'ani', 'becaus', 'befor', 'could', 'doe', 'dure', 'might', 'must', "n't", 'need', 'onc', 'onli', 'ourselv', 'sha', 'themselv', 'veri', 'whi', 'wo', 'would', 'yourselv']
        stop_words.extend(new_stop_words)
        return Pipeline([
            ('columnselector', self.TextSelector(key = 'processed')),
            ('tfidf', tsk.TfidfVectorizer(tokenizer = self.StemTokenizer(), stop_words = stop_words)),
            ('selection', SelectPercentile(chi2, percentile = 50)),
            ('classifier', OneVsRestClassifier(LinearSVC(penalty = 'l2', loss = 'squared_hinge', dual = False, max_iter = 1000, class_weight = 'balanced'), n_jobs = -1))
        ])


class ModelManager(object):
    model_registry = {  # TODO: Add a hook to register user-created models
        'dummy': DummyModel,
        'nb': NaiveBayesModel,
        'logreg': LogisticRegressionModel,
        'fullreport': FullReportModel,
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

    def run_model(self, run_forever=False):
        while True:
            jobs = db_models.DocumentProcessingJob.objects.filter(status='queued').order_by('created_on')
            for job in jobs:
                filename = job.document.docfile.name
                print('Processing Job #%d: %s' % (job.id, filename))
                try:
                    report = self.model.process_job(job)
                    with transaction.atomic():
                        self._save_report(report, job.document)
                        job.delete()
                    print('Created report %s' % report.name)
                except Exception as ex:
                    job.status = 'error'
                    job.message = str(ex)
                    job.save()
                    print(f'Failed to create report for {filename}.')
                    print(traceback.format_exc())

            if not run_forever:
                return
            time.sleep(1)

    def get_model_filepath(self, model_class):
        filepath = settings.ML_MODEL_DIR + '/' + model_class.__name__ + '.pkl'
        return filepath

    def train_model(self):
        self.model.train()
        self.model.test()
        filepath = self.get_model_filepath(self.model.__class__)
        self.model.save_to_file(filepath)
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
        mm = ModelManager(model_key)
        model_name = mm.model.__class__.__name__
        if mm.model.last_trained is None:
            last_trained = 'Never trained'
            trained_techniques_count = 0
        else:
            last_trained = mm.model.last_trained.strftime('%m/%d/%Y %H:%M:%S UTC')
            trained_techniques_count = len(mm.model.detailed_f1_score)

        average_f1_score = round((mm.model.average_f1_score or 0.0) * 100, 2)
        stored_scores = mm.model.detailed_f1_score or []
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
