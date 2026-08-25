# Twitter Sentiment Classification

This project completes the Phase 1 machine-learning assignment from the LLM and AI Agents development plan. It trains and compares Logistic Regression and Linear SVM models on the [Twitter Entity Sentiment Analysis dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis), tunes their hyperparameters, evaluates the selected model, and exposes it through a REST API.

## Dataset and split

The dataset contains a tweet ID, an entity, a sentiment label, and tweet text. The four target classes are `Negative`, `Neutral`, `Positive`, and `Irrelevant`.

The supplied training file contains 74,682 rows and the supplied validation file contains 1,000 rows. The supplied file named validation is treated as the final test split because no other labelled holdout is provided. An internal validation split is created from the training data with `random_state=42`.

Several rows can share an entity and tweet ID while using slightly different wording. Preparation keeps each of these groups in one split so related variants cannot leak between fitting and evaluation. It also removes missing or empty text, repeated normalized text, conflicting labels for the same normalized text, training groups represented in the final test set, and any remaining exact text overlap. The final splits contain:

| Split | Rows | Tweet groups | Purpose |
| --- | ---: | ---: | --- |
| Training | 50,301 | 9,018 | Model fitting and grouped cross-validation |
| Validation | 12,682 | 2,255 | Model comparison and selection |
| Test | 998 | 998 | One final evaluation |

The cleanup removed 2,408 repeated training rows, 379 rows attached to conflicting normalized text, 5,533 training rows from groups represented in the test set, and five remaining exact text overlaps. Two supplied test rows collapsed after normalization. The three saved splits have no entity and tweet-ID group overlap or exact normalized-text overlap.

Text normalization performs the following steps:

- Convert HTML entities and text to lowercase
- Remove URLs, HTML tags, and user mentions
- Keep hashtag words while removing the `#` symbol
- Replace repeated whitespace with a single space
- Tokenize with NLTK's `TweetTokenizer`

The normalized tokens are converted to TF-IDF features using unigrams and bigrams. Features that occur fewer than twice are ignored, very common features are limited with `max_df=0.98`, and the vocabulary is capped at 100,000 features.

## Model selection

Accuracy, macro precision, macro recall, macro F1, weighted F1, classification reports, and confusion matrices are used for evaluation. Macro F1 is the selection metric because it gives equal importance to every sentiment class.

| Model | Validation accuracy | Validation macro F1 |
| --- | ---: | ---: |
| Baseline Logistic Regression | 57.32% | 53.63% |
| Baseline Linear SVM | 54.53% | 51.32% |
| Tuned Logistic Regression | 56.82% | **54.58%** |
| Tuned Linear SVM | **57.36%** | 54.32% |

Hyperparameters are selected with grid search and three-fold stratified group cross-validation. The best settings are:

- Logistic Regression: `C=0.5`, `class_weight="balanced"`
- Linear SVM: `C=0.05`, `class_weight="balanced"`

The best cross-validation macro F1 scores are 52.82% for Logistic Regression and 52.27% for Linear SVM. The tuned Logistic Regression is selected using internal validation macro F1, then refitted on the combined training and validation splits.

## Final test result

The refitted Logistic Regression model achieves 60.02% accuracy and 58.38% macro F1 on the 998-row group-disjoint test split. Per-class F1 ranges from 46.11% for `Irrelevant` to 65.29% for `Positive`.

The model identifies `Negative` and `Positive` tweets most reliably. `Irrelevant` is the weakest class and is often confused with the three sentiment-bearing labels. These scores are substantially lower than row-random results because closely related tweet variants remain together during splitting and cross-validation.

Detailed results are stored in [`reports/results`](reports/results). The final pipeline is generated locally at `models/sentiment_pipeline.joblib` and remains ignored because it is a reproducible model artifact.

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
│   ├── 03_hyperparameter_tuning.ipynb
│   └── 04_model_packaging_and_api_preparation.ipynb
├── reports/results/
├── src/
│   ├── api.py
│   ├── client.py
│   ├── preprocessing.py
│   ├── train.py
│   └── vectorizer.py
└── requirements.txt
```

The numbered notebooks show the complete workflow and should be run in order. Their saved outputs contain the actual split checks, baseline results, tuning results, final test evaluation, example predictions, model reload check, and API contract check.

## Setup and reproduction

Download `twitter_training.csv` and `twitter_validation.csv` from the dataset page and place them in `assignment/data`. From the assignment directory, create and activate a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Start Jupyter from the notebooks directory and run notebooks `00` through `04` in order:

```bash
cd notebooks
jupyter lab
```

The NVIDIA RTX 3050 Ti was detected during verification. Training and inference use CPU because scikit-learn's `LogisticRegression` and `LinearSVC` do not provide CUDA execution.

## Run the API

After running the packaging notebook, start the server from the assignment directory:

```bash
uvicorn src.api:app --reload
```

The API provides:

- `GET /` for a health message
- `POST /predict` for sentiment prediction
- `GET /docs` for interactive API documentation

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

Whitespace-only text is rejected with HTTP 422. With the API running, execute the separate client:

```bash
python src/client.py
```

The verified custom predictions are:

| Text | Prediction |
| --- | --- |
| I absolutely love this new update! | Positive |
| This game is terrible and completely broken. | Negative |
| The maintenance update begins tomorrow. | Neutral |

The examples show the intended positive, negative, and neutral API responses, while the per-class report provides a broader view of the model's limitations.
