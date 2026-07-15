from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.vectorizer import build_tfidf_vectorizer


def build_text_classification_pipeline(classifier: BaseEstimator):
    return Pipeline(
        steps=[
            ("tfidf", build_tfidf_vectorizer()),
            ("classifier", classifier),
        ]
    )


def build_logistic_regression_pipeline():
    classifier = LogisticRegression(
        max_iter=1_000,
        random_state=42,
    )

    return build_text_classification_pipeline(classifier)


def build_svm_pipeline():
    classifier = LinearSVC(
        random_state=42,
    )

    return build_text_classification_pipeline(classifier)


def get_logistic_regression_param_grid():
    return {
        "classifier__C": [
            0.1,
            0.5,
            1.0,
            2.0,
            5.0,
        ],
        "classifier__class_weight": [
            None,
            "balanced",
        ],
    }


def get_svm_param_grid():
    return {
        "classifier__C": [
            0.01,
            0.05,
            0.1,
            0.5,
            1.0,
            2.0,
            5.0,
        ],
        "classifier__class_weight": [
            None,
            "balanced",
        ],
    }