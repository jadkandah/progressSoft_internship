import re
from html import unescape

import pandas as pd
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split

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

    text = unescape(text).lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(r"\1", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def clean_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    remove_exact_duplicates: bool = True,
):
    cleaned_df = df.dropna(subset=[text_column]).copy()
    if remove_exact_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()
    cleaned_df[text_column] = cleaned_df[text_column].apply(normalize_text)
    cleaned_df = cleaned_df[cleaned_df[text_column].str.strip().ne("")]
    return cleaned_df.reset_index(drop=True)


def tokenize_text(text: str):
    return tweet_tokenizer.tokenize(text)


def prepare_dataset_splits(
    training_df: pd.DataFrame,
    supplied_test_df: pd.DataFrame,
    validation_size: float = 0.2,
    random_state: int = 42,
):
    training_clean = clean_dataframe(training_df)
    test_clean = clean_dataframe(supplied_test_df)
    training_rows_after_basic_cleaning = len(training_clean)
    test_rows_after_basic_cleaning = len(test_clean)

    training_clean = training_clean.drop_duplicates(
        subset=["sentiment", "text"]
    ).reset_index(drop=True)
    test_clean = test_clean.drop_duplicates(
        subset=["sentiment", "text"]
    ).reset_index(drop=True)
    duplicate_training_rows_removed = (
        training_rows_after_basic_cleaning - len(training_clean)
    )
    duplicate_test_rows_removed = test_rows_after_basic_cleaning - len(test_clean)

    conflicting_training_texts = (
        training_clean.groupby("text")["sentiment"].nunique()
    )
    conflicting_training_texts = set(
        conflicting_training_texts[conflicting_training_texts > 1].index
    )
    conflicting_training_rows_removed = int(
        training_clean["text"].isin(conflicting_training_texts).sum()
    )
    training_clean = training_clean.loc[
        ~training_clean["text"].isin(conflicting_training_texts)
    ].reset_index(drop=True)

    conflicting_test_texts = (
        test_clean.groupby("text")["sentiment"].nunique()
    )
    conflicting_test_texts = set(
        conflicting_test_texts[conflicting_test_texts > 1].index
    )
    conflicting_test_rows_removed = int(
        test_clean["text"].isin(conflicting_test_texts).sum()
    )
    test_clean = test_clean.loc[
        ~test_clean["text"].isin(conflicting_test_texts)
    ].reset_index(drop=True)

    group_columns = ["entity", "tweet_id"]
    training_group_labels = training_clean.groupby(group_columns)["sentiment"].nunique()
    test_group_labels = test_clean.groupby(group_columns)["sentiment"].nunique()
    if (training_group_labels > 1).any() or (test_group_labels > 1).any():
        raise ValueError("Each tweet group must have exactly one sentiment label")

    test_groups = set(map(tuple, test_clean[group_columns].to_numpy()))
    training_groups = pd.MultiIndex.from_frame(training_clean[group_columns])
    test_group_overlap_mask = training_groups.isin(test_groups)
    test_group_overlap_count = int(test_group_overlap_mask.sum())
    training_clean = training_clean.loc[~test_group_overlap_mask].reset_index(drop=True)

    test_text_overlap_mask = training_clean["text"].isin(test_clean["text"])
    test_text_overlap_count = int(test_text_overlap_mask.sum())
    training_clean = training_clean.loc[~test_text_overlap_mask].reset_index(drop=True)

    group_labels = (
        training_clean.groupby(group_columns, as_index=False)["sentiment"].first()
    )
    train_groups, validation_groups = train_test_split(
        group_labels,
        test_size=validation_size,
        random_state=random_state,
        stratify=group_labels["sentiment"],
    )

    train_group_keys = set(map(tuple, train_groups[group_columns].to_numpy()))
    training_group_index = pd.MultiIndex.from_frame(training_clean[group_columns])
    train_mask = training_group_index.isin(train_group_keys)
    train_clean = training_clean.loc[train_mask].reset_index(drop=True)
    validation_clean = training_clean.loc[~train_mask].reset_index(drop=True)

    summary = {
        "training_rows_after_basic_cleaning": training_rows_after_basic_cleaning,
        "duplicate_training_rows_removed": duplicate_training_rows_removed,
        "conflicting_training_rows_removed": conflicting_training_rows_removed,
        "training_group_overlap_rows_removed": test_group_overlap_count,
        "training_text_overlap_rows_removed": test_text_overlap_count,
        "test_rows_after_basic_cleaning": test_rows_after_basic_cleaning,
        "duplicate_test_rows_removed": duplicate_test_rows_removed,
        "conflicting_test_rows_removed": conflicting_test_rows_removed,
        "train_rows": len(train_clean),
        "validation_rows": len(validation_clean),
        "test_rows": len(test_clean),
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "test_groups": len(test_groups),
    }

    return train_clean, validation_clean, test_clean, summary
