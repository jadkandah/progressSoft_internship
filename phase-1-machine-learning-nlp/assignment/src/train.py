from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.vectorizer import build_tfidf_vectorizer


def build_logistic_regression_pipeline():

    return Pipeline(
        steps=[
            ("tfidf", build_tfidf_vectorizer()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        ]
    )
