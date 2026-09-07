# 08 - Other Layers, Activation Functions, Loss Functions, and Optimizers

## Objective

Compare hidden activation functions and optimizers in the existing IMDB multilayer perceptron while keeping the data pipeline, architecture, training budget, and validation rule controlled.

## Concepts

- LeakyReLU, PReLU, SELU, GELU, Softplus, and Swish
- Activation-aware weight initialization
- Linear layers, batch normalization, and binary logits
- Numerically stable binary cross-entropy
- SGD, momentum, Nesterov momentum, Adam, RMSProp, and AMSGrad
- Validation checkpointing and conditional experimental conclusions

## Tasks

- [x] Reuse the leak-free IMDB split and training-only TF-IDF representation
- [x] Keep the 128/64 MLP architecture and hidden normalization controlled
- [x] Compare LeakyReLU, PReLU, SELU, GELU, Softplus, and Swish
- [x] Match hidden-layer initialization to each activation family
- [x] Select the best activation using validation loss
- [x] Compare SGD, momentum, Nesterov momentum, Adam, RMSProp, and AMSGrad
- [x] Use fixed, declared learning rates and equal training budgets
- [x] Retain the lowest-validation-loss checkpoint within every run
- [x] Explain the stable binary loss and the roles of the surrounding layers
- [x] Evaluate only the final validation-selected checkpoint on the held-out test set
- [x] Record suitable metrics, timings, gradient checks, and limitations
- [x] Detect accelerator availability and document the device choice

## Notes

The experiment reuses the earlier MLP's 29,748/9,916/9,916 stratified train/validation/test split and 20,000-feature L2-normalized TF-IDF representation. Reviews identical after preprocessing remain deduplicated before splitting, and the vocabulary and IDF are learned from training data only.

Every candidate uses the same 128/64 batch-normalized MLP, seed, mini-batch order, four-epoch budget, fused binary-cross-entropy loss, gradient ceiling, and validation-loss checkpointing. Kaiming initialization is used for LeakyReLU and PReLU, LeCun normal initialization for SELU, and Xavier initialization for GELU, Softplus, and Swish. Keeping batch normalization for all candidates isolates the activation change; this is not a test of SELU's separate self-normalizing architecture.

With Adam fixed at `1e-3`, LeakyReLU produced the lowest activation-comparison validation loss, `0.2835`, and validation F1, `0.8896`. GELU was the close runner-up at `0.2859` validation loss. The result is specific to this architecture and four-epoch budget.

The optimizer comparison fixed LeakyReLU and used declared family-appropriate learning rates: `0.1` for SGD, `0.05` for both momentum variants, and `1e-3` for Adam, RMSProp, and AMSGrad. RMSProp selected its first epoch with validation loss `0.2817` and F1 `0.8899`, narrowly ahead of AMSGrad and Adam. Because the learning rates were not tuned separately, this is a comparison of the stated configurations rather than a universal optimizer ranking.

The final LeakyReLU and RMSProp checkpoint achieved held-out accuracy `0.8903`, precision `0.9064`, recall `0.8714`, F1 `0.8885`, and ROC AUC `0.9573`. Its confusion matrix contains 4,491 true negatives, 448 false positives, 640 false negatives, and 4,337 true positives. It remains close to, but does not improve on, the earlier tuned regularized MLP's `0.8920` accuracy and `0.8931` F1; the component comparison is informative even without a new best test score.

All runs remained finite. Vectorization took 17.92 seconds, and the two comparison stages trained in 215.54 seconds. The NVIDIA GeForce RTX 3050 Ti was detected, but the installed PyTorch build exposes no CUDA support, so eight CPU threads were used. Checkpoints stayed in memory and no model artifacts were added.

The executed notebook contains the controlled comparisons, complete validation tables, plots, held-out metrics, confusion matrix, and observations: [`notebooks/activation_optimizer_comparison.ipynb`](notebooks/activation_optimizer_comparison.ipynb).

## Resources

- [PyTorch activation functions](https://docs.pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
- [PyTorch optimization algorithms](https://docs.pytorch.org/docs/stable/optim.html)
- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [PyTorch initialization functions](https://docs.pytorch.org/docs/stable/nn.init.html)
