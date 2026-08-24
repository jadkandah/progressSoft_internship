import argparse
import json

import requests


def predict_sentiment(text, url="http://127.0.0.1:8000/predict"):
    response = requests.post(url, json={"text": text}, timeout=30)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    arguments = parser.parse_args()
    print(json.dumps(predict_sentiment(arguments.text, arguments.url), indent=2))
