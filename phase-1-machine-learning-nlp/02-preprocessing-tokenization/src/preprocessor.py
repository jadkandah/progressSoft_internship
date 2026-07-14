import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

PORTER_STEMMER = PorterStemmer()

def lowercase_text(text: str) -> str:
    """Convert text to lowercase."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return text.lower()


def remove_html_tags(text: str) -> str:
    """Remove HTML tags such as <br /> from text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Remove URLs that begin with http, https, or www."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, " ", text)


def remove_numbers(text: str) -> str:
    """Remove numeric digits from text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return re.sub(r"\d+", " ", text)


def remove_punctuation_and_special_characters(text: str) -> str:
    """
    Remove punctuation and special characters.

    Keeps letters and whitespace only.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return re.sub(r"[^a-zA-Z\s]", " ", text)


def normalize_whitespace(text: str) -> str:
    """
    Convert tabs, new lines, carriage returns, and repeated spaces
    into a single space.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return re.sub(r"\s+", " ", text).strip()

def preprocess_text(text: str) -> str:
    """
    Apply the basic text preprocessing pipeline.

    Processing order:
    1. Convert text to lowercase.
    2. Remove HTML tags.
    3. Remove URLs.
    4. Remove numbers.
    5. Remove punctuation and special characters.
    6. Normalize whitespace.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = lowercase_text(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_numbers(text)
    text = remove_punctuation_and_special_characters(text)
    text = normalize_whitespace(text)

    return text

NEGATION_WORDS = {"no", "not", "nor", "never"}

ENGLISH_STOP_WORDS = set(stopwords.words("english")) - NEGATION_WORDS

def remove_stop_words(
    text: str,
    stop_words: set[str] | None = None,
) -> str:
    """
    Remove common English stop words while preserving negation words.

    Parameters
    ----------
    text:
        Input text that has already been cleaned.

    stop_words:
        Optional custom set of stop words. If not provided,
        the default English NLTK stop-word set is used.

    Returns
    -------
    str
        Text without the selected stop words.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    words_to_remove = (
        ENGLISH_STOP_WORDS if stop_words is None else stop_words
    )

    tokens = text.split()

    filtered_tokens = [
        token
        for token in tokens
        if token not in words_to_remove
    ]

    return " ".join(filtered_tokens)

def preprocess_text(
    text: str,
    remove_stopwords: bool = False,
    apply_stemming: bool = False,
) -> str:
    """
    Apply the text preprocessing pipeline.

    Parameters
    ----------
    text:
        Input text.

    remove_stopwords:
        If True, remove common English stop words while preserving
        negation words such as "not" and "never".
    apply_stemming:
        If True, apply stemming to the text.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = lowercase_text(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_numbers(text)
    text = remove_punctuation_and_special_characters(text)
    text = normalize_whitespace(text)

    if remove_stopwords:
        text = remove_stop_words(text)

    if apply_stemming:
        text = stem_text(text)

    return text

def stem_text(
    text: str,
    stemmer: PorterStemmer | None = None,
) -> str:
    """
    Stem every token in a cleaned string.

    Parameters
    ----------
    text:
        Cleaned text containing whitespace-separated tokens.

    stemmer:
        Optional NLTK stemmer. The Porter stemmer is used by default.

    Returns
    -------
    str
        Text containing stemmed tokens.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    selected_stemmer = PORTER_STEMMER if stemmer is None else stemmer

    tokens = text.split()

    stemmed_tokens = [
        selected_stemmer.stem(token)
        for token in tokens
    ]

    return " ".join(stemmed_tokens)