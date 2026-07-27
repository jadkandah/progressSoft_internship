# Phase 1.3 — Text Statistics

## Objective

Measure and interpret the vocabulary and sequence patterns in the IMDB 50K
Movie Reviews dataset. This part builds directly on the preprocessing and
tokenization choices made in part 02.

## Scope

The notebook follows this learning plan:

- Build a corpus word-frequency distribution.
- Calculate total words, unique words, and the 10 most frequent words.
- Implement contiguous N-grams for any positive value of `n`.
- Find the 10 most frequent bigrams, trigrams, and 4-grams.
- Calculate Type-Token Ratio (TTR).
- Calculate hapax-legomena and dis-legomena proportions.
- Calculate the mean, median, and variance of review lengths.
- Calculate and interpret Shannon entropy for the word distribution.
- Summarize the results and discuss how preprocessing choices affect them.

## Project Structure

```text
03-text-statistics/
├── README.md
└── notebooks/
    └── text_statistics.ipynb
```

No separate Python module is included. The functions in this part are small,
analysis-specific exercises and are kept next to their explanations and results
in the notebook. The notebook reuses the dataset and source modules from
[Phase 1.2 — Text Preprocessing and Tokenization](../02-preprocessing-tokenization/README.md)
instead of duplicating them.

## Suggested Order

1. Review the analysis choices and write your expectations.
2. Load the IMDB reviews and reuse the preprocessing/tokenization code from
   part 02.
3. Complete the frequency and N-gram exercises.
4. Calculate the lexical-diversity and review-length statistics.
5. Calculate Shannon entropy and verify the probability distribution.
6. Summarize and interpret all results in the final section.

## Running the Notebook

Activate the repository virtual environment, start Jupyter, and run
`notebooks/text_statistics.ipynb` from top to bottom. The notebook expects the
IMDB CSV at:

```text
../02-preprocessing-tokenization/data/IMDB Dataset.csv
```

## Completion Criteria

Part 03 is complete when every exercise has an implementation or explanation,
all sanity checks pass, results are clearly labelled, and the final reflection
explains what the statistics mean rather than only listing numbers.

## Next Part

Continue with [Phase 1.4 — Search](../04-search/README.md).
