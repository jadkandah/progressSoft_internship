import re
from html import unescape
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer


# These patterns can be used many times to detect any of the following
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
WHITESPACE_PATTERN = re.compile(r"\s+")

tweet_tokenizer = TweetTokenizer(
    preserve_case=False,
    reduce_len=True,
    strip_handles=True,
)


def normalize_text(text: str):

    if not isinstance(text, str):
        return ""

    text = unescape(text)
    text = text.lower()

    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)

    # Keep the hashtag word but remove the # symbol.
    text = HASHTAG_PATTERN.sub(r"\1", text)

    text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()


def clean_dataframe(df: pd.DataFrame,text_column: str = "text",remove_exact_duplicates: bool = True):

    cleaned_df = df.copy()

    cleaned_df = cleaned_df.dropna(subset=[text_column])

    if remove_exact_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()

    cleaned_df[text_column] = cleaned_df[text_column].apply(normalize_text)

    # Remove rows that became empty after preprocessing.
    cleaned_df = cleaned_df[
        cleaned_df[text_column].str.strip().ne("")
    ]

    return cleaned_df.reset_index(drop=True)

def tokenize_text(text: str):
    return tweet_tokenizer.tokenize(text)

