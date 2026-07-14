from collections.abc import Sequence

import pandas as pd
from sklearn.pipeline import Pipeline


def predict_sentiment(model: Pipeline,texts: Sequence[str]):

    invalid_positions = [
        index
        for index, text in enumerate(texts)
        if not isinstance(text, str) or not text.strip()
    ]

    if invalid_positions:
        raise ValueError(
            "Every input must be a non-empty string. "
            f"Invalid positions: {invalid_positions}"
        )

    predictions = model.predict(list(texts))

    return pd.DataFrame(
        {
            "text": list(texts),
            "predicted_sentiment": predictions,
        }
    )