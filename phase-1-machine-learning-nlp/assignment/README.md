# Twitter Sentiment Classification

This project completes the Phase 1 machine-learning assignment from the LLM and AI Agents development plan. It trains and compares simple machine-learning models on the [Twitter Entity Sentiment Analysis dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis), tunes their hyperparameters, evaluates the selected model, and exposes it through a REST API.

## Dataset

The dataset contains a tweet ID, an entity, a sentiment label, and tweet text. The four target classes are `Negative`, `Neutral`, `Positive`, and `Irrelevant`.

The supplied training set contains 74,682 rows and the supplied validation set contains 1,000 rows. Rows with missing text were removed because the text is the model's only input. Exact duplicates, empty normalized texts, and conflicting entity-text labels were also removed from the training data. The resulting training set contains 69,384 rows. The validation set remains at 1,000 rows.

Text normalization performs the following steps:

- Convert HTML entities and text to lowercase
- Remove URLs, HTML tags, and user mentions
- Keep hashtag words while removing the `#` symbol
- Replace repeated whitespace with a single space
- Tokenize with NLTK's `TweetTokenizer`

The normalized tokens are converted to TF-IDF features using unigrams and bigrams. Features that occur fewer than twice are ignored, very common features are limited with `max_df=0.98`, and the vocabulary is capped at 100,000 features.

## Models and evaluation

Logistic Regression and Linear SVM were used as baseline classifiers. Accuracy, macro precision, macro recall, macro F1, weighted F1, training time, prediction time, classification reports, and confusion matrices were used for evaluation. Macro F1 was the tuning metric because it gives equal importance to every sentiment class.

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Baseline Logistic Regression | 96.8% | 96.74% | 96.80% |
| Baseline Linear SVM | 97.7% | 97.70% | 97.70% |
| Tuned Logistic Regression | 97.5% | 97.56% | 97.51% |
| Tuned Linear SVM | **98.0%** | **98.00%** | **98.00%** |

Hyperparameters were selected with grid search and three-fold stratified cross-validation. The best settings were:

- Logistic Regression: `C=5.0`, `class_weight="balanced"`
- Linear SVM: `C=2.0`, `class_weight=None`

The tuned Linear SVM was selected because it produced the highest validation accuracy and macro F1. It also performed consistently across the four classes. Its cross-validation macro F1 was 92.01%, while its score on the supplied validation set was 98.00%. This difference suggests that the supplied validation set may be easier or more similar to the training data than the cross-validation folds, so the cross-validation result is a more conservative estimate of performance on unseen tweets.

Detailed results are stored in [`reports/results`](reports/results). The final fitted TF-IDF and Linear SVM pipeline is stored in [`models/sentiment_pipeline.joblib`](models/sentiment_pipeline.joblib).

## Project structure

```text
assignment/
├── data/
│   ├── processed/
│   ├── twitter_training.csv
│   └── twitter_validation.csv
├── models/
│   └── sentiment_pipeline.joblib
├── notebooks/
│   ├── 00_data_preparation.ipynb
│   ├── 01_data_inspection.ipynb
│   ├── 02_baseline_model_comparison.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   └── 06_model_packaging_and_api_preparation.ipynb
├── reports/results/
├── src/
│   ├── api.py
│   ├── client.py
│   ├── preprocessing.py
│   ├── train.py
│   └── vectorizer.py
└── requirements.txt
```

The numbered notebooks show the complete workflow and should be read in order. They cover data preparation and inspection, both baseline models, model comparison, hyperparameter tuning, final model selection, packaging, and example predictions.

## Setup

Create and activate a virtual environment from the assignment directory, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the API

Start the server from the assignment directory:

```bash
uvicorn src.api:app --reload
```

The API provides:

- `GET /` for a health message
- `POST /predict` for sentiment prediction
- `GET /docs` for the interactive FastAPI documentation

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I absolutely love this new update!"}'
```

Example response:

```json
{
  "text": "I absolutely love this new update!",
  "predicted_sentiment": "Positive"
}
```

With the API running, execute the separate client to test several examples:

```bash
python src/client.py
```

The saved model predicted the examples used in the project as follows:

| Text | Prediction |
| --- | --- |
| I absolutely love this new update! | Positive |
| This game is terrible and completely broken. | Negative |
| The maintenance update begins tomorrow. | Neutral |
