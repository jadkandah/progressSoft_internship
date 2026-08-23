# Progress Tracker

## Phase 1: Machine Learning & Classical NLP

### 01 - NumPy Primer

Status: Done

* [x] Create NumPy primer directory
* [x] Select and document a numerical dataset
* [x] Inspect dataset rows, columns, dimensions, and data types
* [x] Convert dataset features into a NumPy array
* [x] Practice one-dimensional and multidimensional indexing
* [x] Practice row, column, and range slicing
* [x] Implement reshaping and transposing
* [x] Implement numerical operations
* [x] Implement logical operations
* [x] Implement broadcasting examples
* [x] Compute min, max, mean, and standard deviation using NumPy
* [x] Implement equivalent operations using pure Python
* [x] Validate that NumPy and pure Python results match
* [x] Benchmark both implementations using repeated measurements
* [x] Record benchmark mean and standard deviation
* [x] Explain why NumPy is faster
* [x] Complete README and observations

### 02 - Text Preprocessing & Tokenization

Status: Done

* [x] Create preprocessing and tokenization directory
* [x] Download and document the IMDB 50K Reviews dataset
* [x] Inspect dataset structure and labels
* [x] Implement a basic text preprocessing pipeline
* [x] Implement regex-based cleaning (punctuation, numbers, special characters, URLs, HTML, whitespace)
* [x] Add additional preprocessing features (e.g., stop-word removal, repeated whitespace removal)
* [x] Explain and implement stemming
* [x] Explain what tokenization is and why it is important
* [x] Explain word-level, character-level, and subword tokenization
* [x] Implement a basic tokenizer from scratch
* [x] Compare the custom tokenizer with NLTK and spaCy
* [x] Explain BPE, WordPiece, and SentencePiece
* [x] Complete README and observations

### 03 - Text Statistics

Status: In Progress

* [x] Create a notebook-first text-statistics skeleton
* [ ] Load and prepare the IMDB review corpus
* [ ] Calculate total words, unique words, and the top 10 words
* [ ] Implement contiguous N-grams for any positive N
* [ ] Find the top 10 N-grams for N = 2, 3, and 4
* [ ] Calculate Type-Token Ratio
* [ ] Calculate hapax-legomena and dis-legomena proportions
* [ ] Calculate mean, median, and variance of review lengths
* [ ] Calculate Shannon entropy for the word distribution
* [ ] Add sanity checks and clearly labelled results
* [ ] Complete the final interpretation and README observations

### 04 - Search and Information Retrieval

Status: In Progress

* [x] Create a notebook-first search skeleton
* [x] Explain what indexing is and how documents get indexed
* [x] Explain TF-IDF, implement it, and compute it for each word in one IMDB review
* [ ] Explain how ranking works given any user query
* [ ] Explain how most search engines work
* [x] Use Elasticsearch to index documents and perform a query in Python

### 11 - Support Vector Machines

Status: Done

* [x] Explain how SVM works and how it differs from Logistic Regression
* [x] Reuse the IMDB preprocessing, tokenization, and TF-IDF pipeline
* [x] Apply a very-large-C linear SVM as a hard-margin approximation
* [x] Report hard-margin classification metrics and margin violations
* [x] Use a train/validation/test split for soft-margin tuning
* [x] Tune C using validation F1
* [x] Evaluate the selected soft-margin model on the held-out test set
* [x] Compare hard-margin and soft-margin results


### Assignment Checklist

Status: Paused

#### Project Setup
* [ ] Create the assignment project structure
* [ ] Configure a virtual environment and dependencies
* [ ] Download and inspect the Twitter Entity Sentiment Analysis dataset
#### Data Pipeline
* [ ] Build the preprocessing pipeline
* [ ] Apply tokenization using a standard NLP library
* [ ] Implement TF-IDF vectorization
* [ ] Create reusable preprocessing and inference pipelines
#### Model Development
* [ ] Train a baseline Logistic Regression model
* [ ] Train an SVM model
* [ ] Compare both models
* [ ] Perform hyperparameter tuning
* [ ] Select the final model
#### Model Evaluation
* [ ] Evaluate using Accuracy, Precision, Recall, F1-score, and Confusion Matrix
* [ ] Analyze model strengths and weaknesses
* [ ] Test the model using custom sentiment examples
#### Model Packaging
* [ ] Save the trained model
* [ ] Save the preprocessing and vectorization pipeline
* [ ] Verify that the saved artifacts can be reloaded correctly
#### REST API
* [ ] Build an inference API using FastAPI
* [ ] Validate API input and responses
* [ ] Create a Python client for inference
#### Documentation
* [ ] Write a professional README
* [ ] Organize the repository for GitHub



## Phase 2: Deep Learning

## Phase 3: AI Agents
