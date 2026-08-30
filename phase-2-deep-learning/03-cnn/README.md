# 03 - Convolutional Neural Networks

## Objective

Understand convolutional-network geometry and train a compact PyTorch 1D CNN for IMDB sentiment classification.

## Concepts

- Cross-correlation under the conventional convolution name
- Kernel size, stride, padding, and pooling
- Receptive-field calculation
- One-dimensional and two-dimensional convolution
- Dilated convolution
- Token embeddings and local sequence features

## Tasks

- [x] Explain convolution, kernel size, stride, padding, and pooling
- [x] Explain and calculate receptive fields
- [x] Compare 1D and 2D CNNs
- [x] Explain dilated convolution
- [x] Reuse the IMDB preprocessing and tokenization pipeline
- [x] Create a reproducible, model-visible-disjoint train/validation/test split
- [x] Learn the token vocabulary from training data only
- [x] Implement and train a 1D CNN in PyTorch
- [x] Report suitable held-out classification metrics and learning curves
- [x] Detect accelerator availability and document the device choice

## Notes

The assigned two-dimensional `Conv 3x3 (stride 1) -> MaxPool 2x2 (stride 2) -> Conv 3x3 (stride 1)` stack has an `8x8` receptive field. Along either spatial axis, the field expands from one position to three, four, and then eight, while the effective jump becomes two after pooling. The analogous one-dimensional stack sees eight input positions.

After 423 reviews that were identical after preprocessing and 256-token truncation were removed, the experiment used 29,745 training, 9,916 validation, and 9,916 test reviews. No model-visible sequence was shared across splits, including after unknown-token mapping. A 20,000-token vocabulary was learned only from training data; `25.66%` of test reviews were truncated and `4.30%` of retained test tokens were unknown.

The model uses 64-dimensional embeddings, one width-five convolution with 64 feature maps, ReLU, masked global max pooling, dropout, and a binary output logit. It has 1,300,609 trainable parameters and a five-token local receptive field.

The fixed four-epoch model achieved test accuracy `0.8669`, precision `0.8680`, recall `0.8666`, F1 `0.8673`, and ROC AUC `0.9400`. Validation accuracy reached `0.8670` after epoch 4, while validation loss was lowest after epoch 3 and then remained nearly flat as training loss continued to decrease. The compact CNN remained below the earlier TF-IDF MLP baseline, an honest result that local order-sensitive features did not compensate for the discarded long-range context or the lack of architecture tuning.

The NVIDIA GeForce RTX 3050 Ti was detected, but the installed PyTorch build exposes CPU execution only. Sequence encoding took 1.66 seconds and four training epochs with validation took 84.44 seconds on CPU.

The executed notebook contains the theory, split checks, sequence diagnostics, model checks, learning curves, classification report, confusion matrix, and final observations: [`notebooks/cnn_imdb.ipynb`](notebooks/cnn_imdb.ipynb).

## Resources

- [PyTorch: Conv1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
- [PyTorch: Embedding](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [PyTorch: Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
- [PyTorch: BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
