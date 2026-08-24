from pathlib import Path
import json
import sys
import time

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PHASE_1 = REPOSITORY_ROOT / "phase-1-machine-learning-nlp"
PART_02 = PHASE_1 / "02-preprocessing-tokenization"
PART_06 = PHASE_1 / "06-vectorization"
TASK_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = TASK_ROOT / "models" / "sentiment_pipeline.joblib"
SUMMARY_PATH = TASK_ROOT / "reports" / "results" / "model_summary.json"

sys.path.insert(0, str(PART_02 / "src"))
sys.path.insert(0, str(PART_06 / "src"))

from preprocessor import preprocess_text
from tokenizer import regex_tokenize
from vectorizer import CustomTfidfVectorizer


def train_and_save_model():
    imdb_path = PART_02 / "data" / "IMDB Dataset.csv"
    if not imdb_path.exists():
        raise FileNotFoundError(
            f"Place 'IMDB Dataset.csv' in {imdb_path.parent} before training."
        )

    imdb = pd.read_csv(imdb_path)
    required_columns = {"review", "sentiment"}
    if not required_columns.issubset(imdb.columns):
        raise KeyError(f"The dataset must contain {sorted(required_columns)}.")

    imdb = imdb.dropna(subset=["review", "sentiment"])
    imdb = imdb.drop_duplicates(subset="review").reset_index(drop=True)
    imdb["label"] = imdb["sentiment"].map({"negative": 0, "positive": 1})
    if imdb["label"].isna().any():
        raise ValueError("Sentiment labels must be negative or positive.")

    development_reviews, test_reviews, development_labels, test_labels = (
        train_test_split(
            imdb["review"],
            imdb["label"],
            test_size=0.20,
            random_state=42,
            stratify=imdb["label"],
        )
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                CustomTfidfVectorizer(
                    preprocessor=preprocess_text,
                    tokenizer=regex_tokenize,
                    min_df=5,
                    max_df=0.95,
                    max_features=30_000,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    loss="squared_hinge",
                    dual="auto",
                    tol=1e-4,
                    max_iter=10_000,
                    random_state=42,
                ),
            ),
        ]
    )

    started = time.perf_counter()
    pipeline.fit(development_reviews, development_labels)
    training_seconds = time.perf_counter() - started
    predictions = pipeline.predict(test_reviews)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    summary = {
        "random_seed": 42,
        "development_reviews": int(len(development_reviews)),
        "test_reviews": int(len(test_reviews)),
        "vocabulary_size": int(len(pipeline.named_steps["tfidf"].vocabulary_)),
        "selected_C": 1.0,
        "training_seconds": round(training_seconds, 4),
        "test_accuracy": round(accuracy_score(test_labels, predictions), 6),
        "test_precision": round(
            precision_score(test_labels, predictions, zero_division=0), 6
        ),
        "test_recall": round(
            recall_score(test_labels, predictions, zero_division=0), 6
        ),
        "test_f1": round(f1_score(test_labels, predictions, zero_division=0), 6),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(train_and_save_model(), indent=2))
