from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import normalize_text, tokenize_text


def build_tfidf_vectorizer():
    return TfidfVectorizer(
        preprocessor=normalize_text,
        tokenizer=tokenize_text,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=100_000,
    )
