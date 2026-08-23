# 11 - Support Vector Machines

## Objective

Compare a practical hard-margin approximation with a tuned soft-margin linear SVM on the same IMDB sentiment-classification problem used for Logistic Regression.

## Concepts

- Maximum-margin decision boundaries and support vectors
- Difference between SVM decision scores and Logistic Regression probabilities
- Hard margin, soft margin, margin violations, and the `C` hyperparameter
- Validation-based hyperparameter selection
- Why a hard-margin solution may not exist for noisy real-world text

## Tasks

- [x] Explain how SVM works and how it differs from Logistic Regression
- [x] Apply a very-large-`C` linear SVM as a hard-margin approximation
- [x] Measure classification metrics and remaining margin violations
- [x] Tune soft-margin `C` values with a train/validation/test split
- [x] Select with validation F1 and evaluate once on the test set
- [x] Compare hard-margin and soft-margin behavior

## Notes

The experiment reuses the project's custom preprocessing, regex tokenizer, and TF-IDF vectorizer. After duplicate reviews were removed, the data was split into 29,748 training, 9,917 validation, and 9,917 test reviews. The vectorizer was fitted on training data only and produced 30,000 features.

An exact hard-margin SVM requires linearly separable data. IMDB review language is noisy and overlapping, so the notebook uses `C=1,000,000` as a practical approximation and reports this assumption explicitly. That model achieved perfect training accuracy but raised a convergence warning, left 206 points inside the margin, and reached test F1 `0.8610`.

For soft margin, `C` values `0.01`, `0.1`, `1`, and `10` were compared using validation F1. `C=1` was selected. It achieved test accuracy `0.8921` and test F1 `0.8937`, outperforming the hard-margin approximation and showing why allowing some violations improves generalization on this dataset.

The executed notebook, including tables, classification report, confusion matrix, timings, and observations, is in [`notebooks/svm.ipynb`](notebooks/svm.ipynb).

## Resources

- [scikit-learn: LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- [scikit-learn: SVMs](https://scikit-learn.org/stable/modules/svm.html)
