import asyncio
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import spacy
from tqdm import tqdm
import pickle
import forestci as fci

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import sklearn.ensemble as lm
import scipy.stats as st

from app.objects.c_match import Match
from app.utility.base_world import BaseWorld
from app.objects.c_sentence import Sentence
from app.objects.c_report import Status


class Model(BaseWorld):
    def __init__(self):
        self.log = self.add_service('base_model', self)
        self.name = 'base model'
        self.model = None
        self.rnc = None
        self.classes = None
        self.tfid = None
        self.count_vec = None

    @property
    def unique(self):
        return self.name

    async def create_graph_matricies(self, y, classes):
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

    async def embedding_encode(self, y, model):
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

    async def train_embedder(self, y_embed, y, k=5):
        rnc = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        self.log.debug('Fitting...')
        rnc.fit(y_embed, y)
        return rnc

    async def embedding_decode(self, y_embed, rnc):
        self.log.debug('Decoding embedding')
        predicted_labels = rnc.predict(y_embed)
        self.log.debug('Finished')
        return predicted_labels

    async def extract_X(self, X):
        new_X = await self.remove_stops(X)

        count_vec = CountVectorizer(max_features=2500)
        all_counts = count_vec.fit_transform(new_X)
        self.count_vec = count_vec

        tfid = TfidfTransformer()
        ext_X = tfid.fit_transform(all_counts)
        self.tfid = tfid

        X_train = ext_X.toarray()
        return X_train

    async def extract_y(self, y):
        binarizer = MultiLabelBinarizer()
        y = self.remove_nones(y)
        Y = binarizer.fit_transform(y)
        self.classes = binarizer.classes_
        nodes, edges  = await self.create_graph_matricies(Y, self.classes)
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(nodes)
        nx_graph.add_edges_from(edges)

        N2V = Node2Vec(nx_graph, dimensions=32, walk_length=30, num_walks=300, workers=1)
        n2v = N2V.fit(window=10, min_count=1, batch_words=8)
        return Y, n2v

    async def train(self):
        self.log.debug("Training model...")

        search = await self.get_service('data_svc').locate('search', dict(tag='attack'))
        training_data = await self.get_service('data_svc').locate('search', dict(tag='training_data'))
        reports = await self.get_service('data_svc').locate('reports',dict(status=Status.COMPLETED))

        labels_r, items_r = await self.parse_reports(reports)
        labels_s, items_s = await self.parse_search(search)
        labels_t, items_t = await self.parse_training_data(training_data)
        X = items_s + items_t + items_r
        y = labels_s + labels_t + labels_r

        X_train = await self.extract_X(X)
        ext_y, n2v = await self.extract_y(y)
        new_y = await self.embedding_encode(ext_y, n2v)
        self.rnc = await self.train_embedder(new_y, ext_y)

        self.model = lm.RandomForestRegressor(n_jobs=-1)
        self.log.debug("base_model: fitting regression model")
        self.model.fit(X_train, new_y)
        self.log.debug("base_model: regression model fit")
        self.inbag = fci.calc_inbag(len(X), self.model)

        self.log.debug("base_model: testing model...")
        test = self.model.predict(X_train)
        lab = await self.embedding_decode(test, self.rnc)
        score = f1_score(ext_y, lab, average='weighted')
        self.log.debug("f1 score on training data: {}".format(score))

        await self.get_service('data_svc').store(self)
        with open('data/base_model-v1.0.0.pkl', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.log.debug("Model trained")

    def store(self, ram):
        if not 'model' in ram.keys():
            existing = None
        else:
            existing = self.retrieve(ram['model'], self.unique)
        if not existing:
            if 'model' in ram.keys():
                ram['model'].append(self)
            else:
                ram['model'] = [self]
            return self.retrieve(ram['model'], self.unique)
        existing.update('name', self.name)
        return existing

    async def learn(self, report, tokens):
        model_arr = await self.get_service('data_svc').locate('model', dict(name='base model'))
        inference = model_arr[0]
        full_out, confidence = await inference.train_final(tokens)
        await self.add_matches(full_out, confidence, report, tokens)

    async def add_matches(self, full_out, confidence, report, tokens):
        search = await self.get_service('data_svc').locate('search', dict(tag='attack'))

        for i in range(len(full_out)):
            sen = Sentence(text=tokens[i])
            if len(full_out[i]) > 0:
                for j in full_out[i]:
                    for s in search:
                        if j == s.code:
                            sen.matches.append(Match(model=self.name, search=s, confidence=confidence[i]))
            report.sentences.append(sen)
        report.completed_models += 1

    async def calc_confidence(self, X, y):
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

    async def train_final(self, tokens):
        new_X = await self.remove_stops(tokens)
        all_counts = self.count_vec.transform(new_X)
        ext_X = self.tfid.transform(all_counts)
        output = self.model.predict(ext_X)
        confidence = await self.calc_confidence(ext_X, output)
        decoded_output = await self.embedding_decode(output, self.rnc)
        full_out = []
        for i in decoded_output:
            temp = []
            for j in np.where(i)[0]:
                temp.append(self.classes[j])
            full_out.append(temp)
        return full_out, confidence

    async def parse_reports(self,reports):
        labels, items = [], []
        for r in reports:
            for s in r.sentences:
                for m in s.matches:
                    new_labels, new_items = await self.parse_search([m.search])
                    labels.extend(new_labels)
                    items.extend(new_items)
        return labels, items

    @staticmethod
    async def remove_stops(X):
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
    async def parse_training_data(training_data):
        labels, items = [], []
        for t in training_data:
            labels.append(t.code)
            items.append(t.description)
        return labels, items

    @staticmethod
    async def parse_search(search):
        labels, items = [], []
        for s in search:
            if s.code is None:
                labels.append(['NO_TECHNIQUE'])
            else:
                labels.append([s.code])
            items.append(s.description)
        return labels, items