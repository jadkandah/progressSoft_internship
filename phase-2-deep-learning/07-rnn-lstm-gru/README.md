# 07 - Recurrent Neural Networks, LSTMs, and GRUs

## Objective

Understand recurrent sequence models and train a compact LSTM for IMDB sentiment classification with leak-free data preparation, validation checkpointing, and held-out evaluation.

## Concepts

- Recurrent units and hidden state
- Ordered and variable-length sequence data
- Vanishing and exploding gradients in vanilla RNNs
- LSTM cell state and input, forget, and output gates
- GRU update and reset gates
- Fixed-length encoding and valid-timestep selection
- Gradient clipping and validation-loss checkpointing

## Tasks

- [x] Explain recurrent units and hidden state
- [x] Identify the types of ordered data processed by RNNs
- [x] Explain the main limitations of vanilla RNNs
- [x] Explain how LSTMs and GRUs improve long-range learning
- [x] Reuse the IMDB preprocessing and tokenization pipeline
- [x] Deduplicate model-visible reviews before a reproducible stratified split
- [x] Learn a sequence vocabulary from training data only
- [x] Pad and truncate token sequences with reported coverage diagnostics
- [x] Implement and train an LSTM classifier in PyTorch
- [x] Select the lowest-validation-loss checkpoint before test evaluation
- [x] Report held-out accuracy, precision, recall, F1, ROC AUC, and a confusion matrix
- [x] Record learning curves, gradient clipping, timing, and honest limitations
- [x] Detect accelerator availability and document the device choice

## Notes

The experiment retained 49,577 unique model-visible reviews and used a seeded stratified 29,745/9,916/9,916 train/validation/test split with no overlap. A 20,000-token vocabulary was learned from training reviews only. The 256-token cap matched the earlier CNN experiment, truncated 25.66% of test reviews, and left 4.30% of retained test tokens unknown.

The model uses 64-dimensional embeddings, a single unidirectional LSTM with a 64-dimensional hidden state, 30% dropout, and one output logit. It was trained with AdamW and stable binary cross-entropy for three fixed epochs. Gradient clipping at a total norm of 1.0 affected 308 updates; the maximum pre-clipping norm was 61.7087.

Validation loss selected epoch 3 at 0.3273, with validation accuracy 0.8699 and F1 0.8674. The retained checkpoint achieved held-out accuracy 0.8728, precision 0.8895, recall 0.8525, F1 0.8706, and ROC AUC 0.9458. It slightly exceeded the earlier untuned CNN's 0.8669 accuracy and 0.8673 F1 under the same split and sequence representation, while the recall gap shows that positive reviews were somewhat more likely to be missed.

The NVIDIA GeForce RTX 3050 Ti was detected, but the installed PyTorch build exposes CPU execution only. Three training and validation epochs took 87.49 seconds in the saved run with eight CPU threads. No checkpoints or datasets were added to the repository.

The executed notebook contains the theory, split checks, training-only vocabulary, LSTM implementation, validation checkpointing, learning curves, held-out metrics, confusion matrix, and observations: [`notebooks/lstm_imdb.ipynb`](notebooks/lstm_imdb.ipynb).

## Resources

- [PyTorch LSTM documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [PyTorch GRU documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- [PyTorch gradient clipping documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
