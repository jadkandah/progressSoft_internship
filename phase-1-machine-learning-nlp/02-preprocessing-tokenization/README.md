# Phase 1.2 — Text Preprocessing and Tokenization

## Objective

Build a configurable text-cleaning pipeline, implement basic tokenizers from
scratch, and compare their behavior with tokenizers from NLTK and spaCy. The
examples use reviews from the IMDB 50K Movie Reviews dataset.

Text and corpus statistics are intentionally excluded from this part. They are
covered in [Phase 1.3 — Text Statistics](../03-text-statistics/README.md).

## What Was Completed

- Inspected the IMDB dataset and its sentiment labels.
- Implemented lowercasing and regex-based removal of HTML, URLs, numbers,
  punctuation, and special characters.
- Normalized whitespace and added optional stop-word removal and stemming.
- Combined the cleaning steps in a reusable `preprocess_text` function.
- Implemented whitespace, regex word-level, and character-level tokenizers.
- Compared the custom tokenizers with NLTK and spaCy in the notebook.
- Explained word-, character-, and subword-level tokenization, including BPE,
  WordPiece, and SentencePiece.

## Project Structure

```text
02-preprocessing-tokenization/
├── data/
│   └── data_retreival.ipynb
├── notebooks/
│   └── text_preprocessing.ipynb
└── src/
    ├── preprocessor.py
    └── tokenizer.py
```

The dataset itself is not tracked in Git. Run the retrieval notebook if a local
copy is needed.

## Running the Work

From the repository root, install the dependencies, then open the main notebook
in Jupyter from your development environment:

```bash
pip install -r requirements.txt
```

Run the notebook from top to bottom. The NLTK comparison may download its Punkt
resources, and the spaCy comparison expects the `en_core_web_sm` model to be
installed.

## Key Observations

- Whitespace tokenization is simple and fast, but punctuation remains attached
  to neighboring words.
- Regex tokenization provides more control over what counts as a token, although
  a simple `\w+` pattern discards punctuation and splits contractions.
- Character tokenization avoids out-of-vocabulary words but produces much
  longer token sequences and carries less word-level meaning.
- NLTK offers specialized rule- and regex-based tokenizers for different text
  types, while spaCy provides a fuller language-processing pipeline.
- Preprocessing choices are task-dependent: removing punctuation, stop words,
  or word endings can discard information that is useful for sentiment.

## Next Part

Continue with [Phase 1.3 — Text Statistics](../03-text-statistics/README.md).
