# 06 - Hyperparameter Tuning and Model Selection

## Objective

Use validation checkpointing, a hand-written learning-rate grid, and Optuna to select a regularized IMDB MLP without using the held-out test set for development decisions.

## Concepts

- Train, validation, and test separation
- Checkpoint selection by validation loss
- Learning-rate sensitivity
- Reproducible grid search
- Seeded Bayesian optimization with Optuna
- Validation-based selection before held-out evaluation

## Tasks

- [x] Reuse the regularized IMDB MLP, split, preprocessing, and TF-IDF representation
- [x] Train for 100 epochs and retain the best validation checkpoint
- [x] Explain why the final epoch is not automatically the best model
- [x] Implement a learning-rate grid without a tuning library
- [x] Select the best checkpoint within every grid candidate
- [x] Tune the learning rate again with Optuna
- [x] Keep test labels outside all checkpoint and hyperparameter decisions
- [x] Report held-out accuracy, precision, recall, F1, ROC AUC, and a confusion matrix
- [x] Record learning curves, search results, timings, and honest observations
- [x] Detect accelerator availability and document the device choice

## Notes

The plan refers once to a previous MNIST MLP, but the preceding roadmap classifier is the IMDB MLP. The experiment therefore treats that word as a continuity typo and reuses the existing IMDB task. It retains the same 29,748/9,916/9,916 stratified train/validation/test split, 20,000 training-only TF-IDF features, 128/64 ReLU architecture, batch normalization, 40% dropout, AdamW weight decay of `1e-4`, and gradient clipping at `1.0`.

The `1e-3` model trained for all 100 requested epochs but selected epoch 2, where validation loss was `0.2720`, accuracy was `0.8916`, and F1 was `0.8930`. By epoch 100, training loss had fallen to `0.0014` while validation loss had risen to `1.0094`. The final epoch was therefore substantially more overfit than the retained checkpoint. The 100-epoch run took `487.72` seconds.

The hand-written grid gave every learning rate six epochs and retained the best checkpoint within each run. It selected `1e-2` at epoch 1 with validation loss `0.2611`, validation accuracy `0.8933`, and validation F1 `0.8936`. The deliberately broad grid also showed that a learning rate of `1.0` remained numerically finite but produced a worse best validation loss of `0.3250`.

Eight seeded Optuna trials searched the same learning-rate hyperparameter on a continuous log scale from `1e-5` to `1e-1`, again with six epochs per trial. Three startup trials established an initial sample before TPE guided the later proposals. Optuna selected `3.880e-4` at epoch 2 with validation loss `0.2569`, accuracy `0.8934`, and F1 `0.8948`, improving on the manual grid's validation result.

The Optuna checkpoint was selected before test evaluation. On the held-out test set it achieved accuracy `0.8920`, precision `0.8870`, recall `0.8993`, F1 `0.8931`, and ROC AUC `0.9591`. This improves on the preceding fixed-final-epoch regularized model, which reached accuracy `0.8757`, F1 `0.8742`, and ROC AUC `0.9473`, while preserving the role of the test set as a final report rather than a tuning signal.

The NVIDIA GeForce RTX 3050 Ti was detected, but the installed PyTorch build exposes CPU execution only. Vectorization took `17.23` seconds; the full notebook used eight CPU threads and kept every checkpoint in memory so no model artifacts were added.

The executed notebook contains the split checks, checkpointed training loop, 100-epoch learning curves, manual grid, Optuna study, held-out comparison, confusion matrix, and final observations: [`notebooks/hyperparameter_tuning_imdb.ipynb`](notebooks/hyperparameter_tuning_imdb.ipynb).

## Resources

- [PyTorch: Saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Optuna: Efficient optimization algorithms](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html)
- [Scikit-learn: Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html)
