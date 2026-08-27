# 02 - Multilayer Perceptrons

## Objective

Build a PyTorch multilayer perceptron for IMDB sentiment classification while matching input normalization and weight initialization to ReLU activations.

## Concepts

- Dense layers, hidden representations, and binary logits
- ReLU activations and batch normalization
- Activation-aware Kaiming and Xavier initialization
- Sparse mini-batch training with TF-IDF inputs
- Train, validation, and test separation

## Tasks

- [x] Reuse the IMDB preprocessing, tokenization, and TF-IDF pipeline
- [x] Create a reproducible, model-visible-disjoint stratified train/validation/test split
- [x] L2-normalize sparse TF-IDF inputs
- [x] Normalize hidden pre-activations before ReLU
- [x] Explain why normalization depends on the activation function
- [x] Initialize ReLU layers with Kaiming initialization
- [x] Explain why initialization depends on the activation function
- [x] Train and evaluate the MLP in PyTorch
- [x] Report suitable held-out classification metrics
- [x] Detect accelerator availability and document the device choice

## Notes

After 420 duplicate and preprocessing-equivalent reviews were removed, the experiment used 29,748 training, 9,916 validation, and 9,916 test reviews, with no model-visible text shared across splits. The custom Phase 1 pipeline produced 20,000 L2-normalized TF-IDF features, with the vocabulary and IDF learned from training data only. Sparse batches remain sparse through the first linear projection, avoiding a dense copy of the full feature matrix.

The model has two ReLU hidden layers with 128 and 64 units. Batch normalization controls the scale entering each ReLU. Kaiming initialization is used for both hidden layers because it preserves variance after ReLU removes negative activations; the final linear logit layer uses Xavier initialization.

The fixed eight-epoch model achieved test accuracy `0.8791`, precision `0.8854`, recall `0.8720`, F1 `0.8786`, and ROC AUC `0.9508`. Validation accuracy peaked at `0.8939` after epoch 1, then declined while training loss approached zero. This is an intentionally unregularized baseline, and the widening validation gap provides a useful comparison for the later regularization and model-selection experiments.

The NVIDIA GeForce RTX 3050 Ti was detected, but the installed PyTorch build exposes CPU execution only. CPU training remained practical because the first projection uses sparse matrix multiplication: vectorization took 17.16 seconds and eight training epochs with validation took 32.70 seconds.

The executed notebook contains the theory, split checks, normalization and initialization checks, learning curves, classification report, confusion matrix, and final observations: [`notebooks/mlp_imdb.ipynb`](notebooks/mlp_imdb.ipynb).

## Resources

- [PyTorch: Build the neural network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [PyTorch: Weight initialization](https://docs.pytorch.org/docs/stable/nn.init.html)
- [PyTorch: BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
- [PyTorch: BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
