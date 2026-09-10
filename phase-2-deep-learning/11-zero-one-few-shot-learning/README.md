# 11 - Zero-Shot, One-Shot, and Few-Shot Learning

## Objective

Understand how the number of labeled examples changes a learning problem, then compare tiny-support KNN classification on frozen BERT embeddings with zero- to three-shot GPT-2 prompting for IMDB sentiment.

## Concepts

- Zero-shot, one-shot, and few-shot learning
- Examples per class and balanced support sets
- Cosine-distance nearest-neighbor classification
- Episodic evaluation and support-set variability
- In-context demonstrations
- Causal language-model label scoring
- Prompt sensitivity and held-out evaluation

## Tasks

- [x] Explain zero-shot, one-shot, and few-shot learning
- [x] Treat each shot as one labeled example per sentiment class
- [x] Reload the serialized BERT Tiny embeddings from the previous task
- [x] L2-normalize embeddings and fix a one-nearest-neighbor cosine classifier
- [x] Compare 1, 3, and 5 labeled reviews per class
- [x] Repeat the KNN experiment across 100 seeded nested support sets
- [x] Report support-set variability on validation data
- [x] Evaluate one predeclared KNN episode on the held-out test embeddings
- [x] Reconstruct the previous task's model-visible IMDB split
- [x] Build balanced nested GPT-2 prompts with 1, 2, and 3 examples per class
- [x] Include a GPT-2 zero-shot baseline
- [x] Score the single-token negative and positive labels directly
- [x] Test the prompts on balanced held-out IMDB reviews
- [x] Record metrics, timing, limitations, and the accelerator decision

## Notes

The shot count is interpreted per class. A one-shot binary experiment therefore uses one negative and one positive example. This keeps both labels represented and makes the KNN and prompting comparisons consistent.

The KNN stage reloads the previous heading's ignored 5.21 MiB pickle containing 6,000 training, 1,500 validation, and 3,000 test BERT Tiny embeddings. Vectors are L2-normalized before a fixed one-nearest-neighbor classifier uses cosine distance. Every one of 100 seeded episodes samples five training examples per class, then uses nested prefixes of 1, 3, and 5. The validation set measures sampling variability without changing the classifier or support examples.

Mean validation accuracy increased from `0.5286` at one shot per class to `0.5399` at three and `0.5429` at five. Standard deviations were `0.0320`, `0.0315`, and `0.0301`, and the distributions overlapped substantially. The predeclared seed-42 episode achieved held-out test accuracy `0.5363`, `0.6023`, and `0.5897`, respectively. The five-shot support did not beat three shots in that individual draw, despite the increasing repeated-validation mean.

The GPT-2 stage recreates the exact model-visible split from the transfer-learning task. It draws three short demonstrations per class from the 6,000-review training subset and six short reviews per class from the untouched 3,000-review test subset. Demonstrations are nested across prompt sizes. Plain GPT-2 is a causal language model rather than an instruction-tuned classifier, so the experiment compares its next-token logits for the single-token labels `negative` and `positive` instead of generating unrestricted text.

On the fixed 12-review prompt set, zero-, one-, two-, and three-shot accuracy was `0.5833`, `0.4167`, `0.7500`, and `0.7500`. Macro F1 was `0.4958`, `0.2941`, `0.7333`, and `0.7333`. The small sample demonstrates prompt behavior rather than full-corpus performance. Results can change with the selected examples, their order, prompt wording, and the label tokens; adding demonstrations did not improve performance monotonically.

The NVIDIA GeForce RTX 3050 Ti was detected, but the installed PyTorch build exposes CPU execution only. The run used eight CPU threads. One hundred KNN validation episodes took `0.76` seconds, exact split reconstruction and sampling took `31.55` seconds, GPT-2 loading from the external cache took `0.94` seconds, and prompt scoring took `17.52` seconds. No new model artifact was written to the repository.

The executed notebook contains the definitions, artifact checks, repeated KNN study, held-out metrics, GPT-2 prompt construction, constrained label scoring, plots, review-level examples, and observations: [`notebooks/shot_learning_imdb.ipynb`](notebooks/shot_learning_imdb.ipynb).

## Resources

- [GPT-2 model card](https://huggingface.co/gpt2)
- [Hugging Face causal language modeling guide](https://huggingface.co/docs/transformers/tasks/language_modeling)
- [scikit-learn nearest neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [BERT Tiny model card](https://huggingface.co/prajjwal1/bert-tiny)
