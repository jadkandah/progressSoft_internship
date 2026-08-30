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

### 12 - FastAPI Serving and Concurrency Experiments

Status: Done

* [x] Refit and package the tuned IMDB sentiment pipeline
* [x] Reuse the existing preprocessing, tokenization, and TF-IDF code
* [x] Build health and prediction endpoints with FastAPI
* [x] Validate input, health responses, and sentiment predictions
* [x] Create a separate Python inference client
* [x] Send 1,000 requests at each parallelism level from 5 through 100
* [x] Record completion time, throughput, mean latency, and p95 latency
* [x] Plot completion time and throughput against parallel requests
* [x] Identify and explain the maximum-throughput setting


### Assignment Checklist

Status: Done

#### Project Setup
* [x] Create the assignment project structure
* [x] Configure a virtual environment and dependencies
* [x] Download and inspect the Twitter Entity Sentiment Analysis dataset
#### Data Pipeline
* [x] Build the preprocessing pipeline
* [x] Apply tokenization using a standard NLP library
* [x] Implement TF-IDF vectorization
* [x] Create reusable preprocessing and inference pipelines
* [x] Create reproducible group-disjoint train, validation, and test splits
#### Model Development
* [x] Train a baseline Logistic Regression model
* [x] Train an SVM model
* [x] Compare both models
* [x] Perform hyperparameter tuning
* [x] Select the final model
#### Model Evaluation
* [x] Evaluate using Accuracy, Precision, Recall, F1-score, and Confusion Matrix
* [x] Analyze model strengths and weaknesses
* [x] Test the model using custom sentiment examples
#### Model Packaging
* [x] Save the trained model
* [x] Save the preprocessing and vectorization pipeline
* [x] Verify that the saved artifacts can be reloaded correctly
#### REST API
* [x] Build an inference API using FastAPI
* [x] Validate API input and responses
* [x] Create a Python client for inference
#### Documentation
* [x] Write a professional README
* [x] Organize the repository for GitHub



## Phase 2: Deep Learning

### 01 - Deep Learning Frameworks Primer

Status: Done

* [x] Explain computation graphs and data dependencies
* [x] Explain backpropagation and reverse-mode automatic differentiation
* [x] Describe the responsibilities of a deep learning framework
* [x] Create TensorFlow and PyTorch tensors from lists of floats
* [x] Compare four mathematical tensor operations
* [x] Implement gradient descent without numerical libraries
* [x] Optimize the same scalar function with TensorFlow and PyTorch
* [x] Verify convergence against the analytical minimum
* [x] Detect accelerator availability and document the device choice

### 02 - Multilayer Perceptrons

Status: Done

* [x] Reuse the IMDB preprocessing, tokenization, and TF-IDF pipeline
* [x] Deduplicate model-visible reviews before a reproducible stratified train/validation/test split
* [x] Learn the TF-IDF vocabulary and IDF from training only, then L2-normalize every split
* [x] Explain why normalization must suit the activation function
* [x] L2-normalize sparse inputs and batch-normalize hidden ReLU inputs
* [x] Explain why initialization must suit the activation function
* [x] Apply Kaiming initialization to ReLU layers and Xavier initialization to the output
* [x] Train a sparse-input MLP in PyTorch
* [x] Report held-out accuracy, precision, recall, F1, ROC AUC, and a confusion matrix
* [x] Record learning curves and honest overfitting observations
* [x] Detect accelerator availability and document the device choice

### 03 - Convolutional Neural Networks

Status: Done

* [x] Explain convolution, kernel size, stride, padding, and pooling
* [x] Explain receptive fields and calculate the assigned network's field
* [x] Compare 1D and 2D convolutional networks
* [x] Explain dilated convolution
* [x] Reuse the IMDB preprocessing and tokenization pipeline
* [x] Deduplicate model-visible reviews before a reproducible stratified split
* [x] Learn a sequence vocabulary from training data only
* [x] Pad and truncate token sequences with reported coverage diagnostics
* [x] Implement and train a 1D CNN in PyTorch
* [x] Report held-out accuracy, precision, recall, F1, ROC AUC, and a confusion matrix
* [x] Record learning curves and honest generalization observations
* [x] Detect accelerator availability and document the device choice

## Phase 3: AI Agents
