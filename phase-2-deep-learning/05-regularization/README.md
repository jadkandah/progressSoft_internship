# 05 - Numerical Stability and Neural-Network Regularization

## Objective

Explain vanishing and exploding gradients, then compare the original IMDB MLP with a fixed configuration that adds numerical-stability safeguards and regularization.

## Concepts

- Products of derivatives and weight matrices during backpropagation
- Finite-precision underflow, overflow, and stable fused losses
- Activation-aware initialization and hidden normalization
- Gradient clipping and finite-value checks
- Dropout and decoupled weight decay
- Regularization strength versus model selection

## Tasks

- [x] Explain vanishing and exploding gradients and their numerical consequences
- [x] List the main techniques for avoiding unstable gradients
- [x] List the main neural-network regularization techniques
- [x] Reuse the original IMDB preprocessing, tokenization, TF-IDF, split, and MLP widths
- [x] Keep activation-aware initialization, input normalization, and batch normalization
- [x] Use a numerically stable fused binary-classification loss
- [x] Add dropout, decoupled weight decay, and gradient clipping
- [x] Verify finite losses and gradients and record gradient norms
- [x] Compare the fixed final epoch with the original MLP on the held-out test set
- [x] Report suitable classification metrics and honest generalization observations
- [x] Detect accelerator availability and document the device choice

## Notes

The experiment reused the original MLP's 29,748/9,916/9,916 stratified train/validation/test split after removing 420 duplicate or preprocessing-equivalent reviews. No model-visible review crosses a split. The custom Phase 1 pipeline again learned 20,000 L2-normalized TF-IDF features from training data only.

The architecture retains 128- and 64-unit ReLU hidden layers, Kaiming initialization, batch normalization, a stable binary-cross-entropy-with-logits loss, a batch size of 512, a learning rate of `1e-3`, and a fixed eight-epoch budget. The regularized version adds 40% dropout after each hidden activation, AdamW weight decay of `1e-4` on weight matrices only, and gradient clipping at a total norm of `1.0`.

Every loss and gradient remained finite. The largest pre-clipping gradient norm was `3.6659`, and 63 of 472 updates exceeded the ceiling and were clipped. Final training loss was `0.0046`, compared with `0.0009` for the original MLP, so dropout and weight decay slowed memorization.

The fixed final epoch reached test accuracy `0.8757`, precision `0.8879`, recall `0.8610`, F1 `0.8742`, and ROC AUC `0.9473`. The original fixed-eight-epoch model reached accuracy `0.8791`, F1 `0.8786`, and ROC AUC `0.9508`; regularization therefore lowered final-epoch test accuracy by `0.0034` in this configuration.

Regularization did not eliminate overfitting. Validation loss was lowest at `0.2720` after epoch 2 and rose to `0.5205` by epoch 8, while validation accuracy peaked at `0.8942` after epoch 1. That peak was slightly above the original model's `0.8939`, showing why the following hyperparameter-tuning and model-selection task should retain the best validation checkpoint rather than assume the last epoch is best.

The NVIDIA GeForce RTX 3050 Ti was detected, but the installed PyTorch build exposes CPU execution only. Vectorization took 17.85 seconds and eight training epochs with validation took 37.22 seconds on CPU.

The executed notebook contains the theory, leakage checks, regularized implementation, gradient diagnostics, learning curves, held-out comparison, confusion matrix, and final observations: [`notebooks/regularized_mlp_imdb.ipynb`](notebooks/regularized_mlp_imdb.ipynb).

## Resources

- [PyTorch: Numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [PyTorch: Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
- [PyTorch: AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [PyTorch: Gradient clipping](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- [PyTorch: BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
