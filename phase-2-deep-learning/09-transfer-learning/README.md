# 10 - Pre-trained Models, Transfer Learning, and Fine-Tuning

## Objective

Understand common language-model pretraining objectives, compare random initialization with partial BERT fine-tuning, and reuse frozen BERT representations for classical sentiment classification.

## Concepts

- Pretraining, transfer learning, and fine-tuning
- Sequence-to-sequence, causal language modeling, masked language modeling, and span corruption
- BERT masked-language-model pretraining
- Random initialization and pretrained weights
- Selective layer freezing
- Contextual token embeddings and mean pooling
- Validation checkpointing and held-out evaluation

## Tasks

- [x] Distinguish pretraining, transfer learning, and fine-tuning
- [x] Explain and exemplify four common pretraining objectives
- [x] Identify BERT's masked-language-model objective
- [x] Reuse the IMDB preprocessing pipeline and BERT WordPiece tokenization
- [x] Deduplicate model-visible reviews before a reproducible stratified split
- [x] Train a randomly initialized BERT Tiny sentiment classifier
- [x] Load pretrained BERT Tiny and freeze its embedding parameters
- [x] Fine-tune the final two Transformer layers and a new classification head
- [x] Retain the lowest-validation-loss checkpoint for both neural models
- [x] Extract mean-pooled embeddings with a frozen pretrained encoder
- [x] Serialize and reload the embeddings from an ignored pickle file
- [x] Select a linear SVM regularization value using validation data
- [x] Report held-out accuracy, precision, recall, F1, ROC AUC, and confusion matrices
- [x] Record timing, truncation, limitations, and the accelerator decision

## Notes

The source plan's BERT-on-CIFAR-10 instruction is treated as a dataset typo because BERT accepts token sequences and CIFAR-10 contains images. All three experiments therefore use IMDB sentiment data, which permits a technically valid comparison and continues the surrounding NLP work.

Reviews were preprocessed with the shared Phase 1 pipeline and encoded with the uncased BERT tokenizer. After 436 reviews that were identical at the model-visible 96-token representation were removed, a seeded stratified split was created before fixed 6,000/1,500/3,000 train/validation/test subsets were drawn. No model-visible representation overlaps the splits. The compact token limit makes the CPU experiment practical but truncates 90.67% of sampled test reviews.

Both neural classifiers use the two-layer, 128-hidden-unit BERT Tiny architecture and a new binary classification head. The random model trains all 4,386,049 parameters at a learning rate of `5e-4`. Partial fine-tuning freezes token and position embeddings and updates the final two Transformer layers plus the head, leaving 396,673 parameters trainable at `2e-4`. Both use AdamW, stable binary cross-entropy, gradient clipping at `1.0`, three epochs, and minimum validation loss for checkpoint selection.

The random model selected epoch 2 and achieved held-out accuracy `0.7753`, precision `0.8296`, recall `0.6952`, F1 `0.7565`, and ROC AUC `0.8743`. Partial fine-tuning selected epoch 3 and achieved accuracy `0.7480`, precision `0.7757`, recall `0.7005`, F1 `0.7362`, and ROC AUC `0.8379`. Pretraining did not outperform random initialization under this restricted setup. Frozen embeddings, few updates, and severe truncation limit adaptation, so this result should not be generalized to full-data BERT training.

A separate fully frozen pretrained encoder mean-pooled the token outputs into 128-dimensional review embeddings. The notebook serialized and reloaded 5.21 MiB of embeddings from an ignored pickle file. Validation F1 selected a linear SVM with `C=0.1`; after refitting on training plus validation embeddings, it achieved test accuracy `0.7167`, precision `0.7303`, recall `0.6906`, F1 `0.7099`, and ROC AUC `0.7862`.

The NVIDIA GeForce RTX 3050 Ti was detected, but the installed PyTorch build exposes CPU execution only. Neural training and validation took 180.98 seconds with eight CPU threads, frozen embedding extraction took 8.94 seconds, and the final SVM fit took 0.62 seconds. Neural checkpoints remained in memory, and the serialized embedding file is ignored by Git.

The executed notebook contains the theory, technical assumption, leak-free split, random and pretrained BERT training, validation checkpointing, embedding round trip, SVM selection, learning curves, confusion matrices, and observations: [`notebooks/bert_transfer_learning_imdb.ipynb`](notebooks/bert_transfer_learning_imdb.ipynb).

## Resources

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Hugging Face BERT documentation](https://huggingface.co/docs/transformers/model_doc/bert)
- [BERT Tiny model card](https://huggingface.co/prajjwal1/bert-tiny)
- [scikit-learn LinearSVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
