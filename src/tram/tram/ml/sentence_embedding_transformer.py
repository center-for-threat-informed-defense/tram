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
