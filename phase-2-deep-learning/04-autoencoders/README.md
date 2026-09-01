# 04 - Autoencoders

## Objective

Train a compact PyTorch autoencoder on training-only IMDB word-context profiles, use its bottleneck as word embeddings, and evaluate mean-pooled embeddings with a classical sentiment classifier.

## Concepts

- Encoders, bottlenecks, decoders, and reconstruction objectives
- Distributional word representations
- Review-level word co-occurrence and positive pointwise mutual information
- Training-only representation learning
- Mean-pooled word embeddings for document classification

## Tasks

- [x] Explain autoencoders and bottleneck representations
- [x] Reuse the IMDB preprocessing and tokenization pipeline
- [x] Create a reproducible, model-visible-disjoint train/validation/test split
- [x] Learn the target and context vocabularies from training data only
- [x] Build normalized PPMI word-context profiles without validation or test leakage
- [x] Train a PyTorch autoencoder to reconstruct word profiles
- [x] Extract word embeddings from the bottleneck
- [x] Mean-pool word embeddings into review features
- [x] Train a Logistic Regression sentiment classifier
- [x] Report held-out accuracy, precision, recall, F1, ROC AUC, and a confusion matrix
- [x] Report vocabulary coverage, reconstruction curves, and honest limitations
- [x] Detect accelerator availability and document the device choice

## Notes

After 420 duplicate and preprocessing-equivalent reviews were removed, the experiment used 29,748 training, 9,916 validation, and 9,916 test reviews, with no model-visible review shared across splits. The 8,000 most frequent training words became embedding targets, and the 2,000 most frequent formed the dimensions of review-level co-occurrence profiles. Positive PMI removed associations no stronger than chance, self-context was removed, and every profile was L2-normalized.

The autoencoder maps each 2,000-dimensional profile through a 256-unit ReLU layer to a 64-dimensional linear bottleneck, then mirrors the encoder to reconstruct the profile. Its 1,059,344 parameters were trained for a fixed twelve epochs. Mean squared reconstruction error finished at `0.000283` on the training word rows and `0.000284` on held-out word rows.

Each review was represented by the mean of its in-vocabulary bottleneck vectors. The 8,000-word training vocabulary retained `91.45%` of test tokens. A fixed Logistic Regression classifier trained on these 64-dimensional review features achieved test accuracy `0.8639`, precision `0.8579`, recall `0.8734`, F1 `0.8656`, and ROC AUC `0.9344`.

The result is useful but remains below the earlier TF-IDF MLP baseline. Mean pooling discards word order and negation structure, rare words are ignored, and review-level co-occurrence mixes broad topic with sentiment. The close training and validation reconstruction losses show that the autoencoder generalized across word profiles, but reconstruction quality alone does not guarantee a task-optimal sentiment representation.

The NVIDIA GeForce RTX 3050 Ti was detected, but the installed PyTorch build exposes CPU execution only. Building the PPMI profiles took 4.03 seconds, and twelve autoencoder epochs with validation took 5.90 seconds on CPU.

The executed notebook contains the theory, leakage checks, PPMI construction, reconstruction curves, word-neighbor examples, classification report, confusion matrix, and final observations: [`notebooks/word_autoencoder_imdb.ipynb`](notebooks/word_autoencoder_imdb.ipynb).

## Resources

- [PyTorch: Building models with `nn.Module`](https://docs.pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html)
- [PyTorch: Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [PyTorch: MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)
- [scikit-learn: LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
