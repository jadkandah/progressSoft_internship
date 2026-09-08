# 09 - Brief Introduction to Transformer Networks

## Objective

Understand token embeddings and self-attention, then compare an LSTM, CLASS-pooled self-attention, and the same attention model with learned positional embeddings on IMDB sentiment classification.

## Concepts

- Trainable token embeddings and padding rows
- Queries, keys, values, and scaled dot-product attention
- Multiple attention heads
- Padding masks
- CLASS-token pooling
- Learned positional embeddings
- Residual connections and layer normalization
- Validation-loss checkpointing

## Tasks

- [x] Explain what `torch.nn.Embedding` does
- [x] Reuse the IMDB preprocessing and tokenization pipeline
- [x] Deduplicate model-visible reviews before a reproducible stratified split
- [x] Learn a sequence vocabulary from training data only
- [x] Train and evaluate an embedding-plus-LSTM classifier
- [x] Implement multi-head self-attention from query, key, and value projections
- [x] Prepend a dedicated CLASS token to every attention input
- [x] Mask padding and verify attention probabilities
- [x] Classify from the final contextualized CLASS representation
- [x] Train attention models without and with learned positional embeddings
- [x] Compare positional encoding with all other attention settings controlled
- [x] Retain the lowest-validation-loss checkpoint for each model
- [x] Report held-out accuracy, precision, recall, F1, ROC AUC, and confusion matrices
- [x] Record timing, gradient checks, limitations, and the accelerator decision

## Notes

The experiment retained 49,572 unique model-visible reviews and used a seeded stratified 29,742/9,915/9,915 train/validation/test split with no overlap. A 20,000-token vocabulary, including padding, unknown, and CLASS tokens, was learned from training reviews only. The resource-conscious 128-token cap truncated 66.59% of test reviews and left 4.11% of retained test tokens unknown.

All three models use 32-dimensional token embeddings, one sequence layer, 20% dropout, one output logit, AdamW, stable binary cross-entropy, gradient clipping at 1.0, and the same three-epoch budget. The LSTM classifies from its final valid output. Both attention models prepend CLASS, apply a manually implemented four-head scaled dot-product attention block with a residual connection and layer normalization, and classify from the contextualized CLASS output at position 0. Their shared layers begin with identical weights, and the training random state is reset for a controlled positional comparison. Padding checks confirmed zero attention probability on padded keys and normalized attention rows.

The LSTM selected epoch 2 and achieved held-out accuracy 0.8166, precision 0.8967, recall 0.7172, F1 0.7970, and ROC AUC 0.9193. Self-attention without positions selected epoch 1 and achieved accuracy 0.8453, precision 0.8478, recall 0.8430, F1 0.8454, and ROC AUC 0.9246.

Learned positional embeddings also selected epoch 1. They improved validation loss from 0.3376 to 0.3352 and validation F1 from 0.8511 to 0.8552. On the held-out test set, accuracy increased to 0.8485, F1 to 0.8488, and ROC AUC to 0.9270. Positional information made order available and produced a small improvement in this compact three-epoch experiment.

The NVIDIA GeForce RTX 3050 Ti was detected, but the installed PyTorch build exposes CPU execution only. Training and validation for all three models took 433.73 seconds with eight CPU threads. Checkpoints remained in memory, and no model artifacts or datasets were added.

The executed notebook contains the embedding explanation, leak-free sequence preparation, LSTM baseline, self-attention implementation, CLASS-token and padding checks, positional comparison, learning curves, held-out metrics, confusion matrices, and observations: [`notebooks/transformer_intro_imdb.ipynb`](notebooks/transformer_intro_imdb.ipynb).

## Resources

- [PyTorch Embedding documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [PyTorch softmax documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
- [PyTorch LayerNorm documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
