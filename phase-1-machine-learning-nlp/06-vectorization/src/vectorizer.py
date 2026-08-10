"""Reusable sparse form of the custom TF-IDF vectorizer from week 06."""

from collections import Counter
import math

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CustomTfidfVectorizer(BaseEstimator, TransformerMixin):
    """Apply the project's TF = count/length and IDF = log(N/DF) formulas."""

    def __init__(
        self,
        preprocessor,
        tokenizer,
        min_df=5,
        max_df=0.95,
        max_features=30_000,
    ):
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features

    def _prepare_tokens(self, document):
        cleaned = self.preprocessor(
            str(document), remove_stopwords=False, apply_stemming=False
        )
        return [
            token
            for token in self.tokenizer(cleaned)
            if token.isalpha() and len(token) >= 2
        ]

    def fit(self, documents, y=None):
        documents = list(documents)
        if not documents:
            raise ValueError("At least one training document is required.")

        term_frequencies = Counter()
        document_frequencies = Counter()
        for document in documents:
            tokens = self._prepare_tokens(document)
            term_frequencies.update(tokens)
            document_frequencies.update(set(tokens))

        number_of_documents = len(documents)
        minimum_count = (
            math.ceil(self.min_df * number_of_documents)
            if isinstance(self.min_df, float)
            else self.min_df
        )
        maximum_count = (
            math.floor(self.max_df * number_of_documents)
            if isinstance(self.max_df, float)
            else self.max_df
        )
        terms = [
            term
            for term, frequency in document_frequencies.items()
            if minimum_count <= frequency <= maximum_count
        ]
        terms.sort(key=lambda term: (-term_frequencies[term], term))
        if self.max_features is not None:
            terms = terms[: self.max_features]
        if not terms:
            raise ValueError("The document-frequency settings removed every feature.")

        self.feature_names_ = np.asarray(terms, dtype=object)
        self.vocabulary_ = {
            term: index for index, term in enumerate(self.feature_names_)
        }
        self.idf_ = np.asarray(
            [
                math.log(number_of_documents / document_frequencies[term])
                for term in self.feature_names_
            ],
            dtype=np.float64,
        )
        return self

    def transform(self, documents):
        check_is_fitted(self, attributes=["vocabulary_", "idf_"])
        documents = list(documents)
        rows, columns, values = [], [], []

        for row, document in enumerate(documents):
            tokens = self._prepare_tokens(document)
            if not tokens:
                continue
            counts = Counter(token for token in tokens if token in self.vocabulary_)
            for term, count in counts.items():
                column = self.vocabulary_[term]
                rows.append(row)
                columns.append(column)
                values.append((count / len(tokens)) * self.idf_[column])

        return csr_matrix(
            (values, (rows, columns)),
            shape=(len(documents), len(self.vocabulary_)),
            dtype=np.float64,
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, attributes=["feature_names_"])
        return self.feature_names_.copy()
