import requests


API_URL = "http://127.0.0.1:8000/predict"


def predict_sentiment(text: str) -> dict:
    """
    Send text to the sentiment API and return the JSON response.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string.")

    response = requests.post(
        API_URL,
        json={"text": text},
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


if __name__ == "__main__":
    examples = [
        "I absolutely love this new update!",
        "This game is terrible and completely broken.",
        "The maintenance update begins tomorrow.",
    ]

    for text in examples:
        result = predict_sentiment(text)

        print(f"Text: {result['text']}")
        print(
            f"Predicted sentiment: "
            f"{result['predicted_sentiment']}"
        )
        print("-" * 60)