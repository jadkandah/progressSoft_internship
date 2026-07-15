"""TODO: Implement every required IMDB text statistic.

CHECKLIST
---------
[ ] Load the review column from the IMDB CSV.
[ ] Preprocess and tokenize every review consistently.
[ ] Count total and unique words.
[ ] Find the 10 most frequent words.
[ ] Implement contiguous N-grams for any positive N.
[ ] Find the top 10 N-grams for N = 2, 3, and 4.
[ ] Compute Type-Token Ratio.
[ ] Compute the hapax-legomena proportion.
[ ] Compute the dis-legomena proportion.
[ ] Compute mean, median, and variance of words per review.
[ ] Compute Shannon entropy for the word distribution.
[ ] Return or print clearly labelled results.
[ ] Interpret the results in the notebook.
"""

# TODO: Add imports only when you need them.


def load_imdb_reviews(csv_path: str) -> list[str]:
    """TODO: Load and return the review strings from the CSV."""
    raise NotImplementedError


def extract_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """TODO: Return every contiguous sequence of n tokens."""
    raise NotImplementedError


def corpus_statistics(documents: list[str]) -> dict[str, object]:
    """TODO: Calculate and return every statistic in the checklist."""
    raise NotImplementedError


def main() -> None:
    """TODO: Run the analysis using the IMDB CSV path."""
    pass


if __name__ == "__main__":
    main()
