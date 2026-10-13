"""ML detector: TF-IDF (1-3 word n-grams) + logistic regression, 3 classes:
safe, prompt_injection, harmful_content.

The risk score is the probability that the prompt is NOT safe. (Earlier this
used the confidence of whichever class won, so a prompt the model was 95%
sure was *safe* scored 0.95 risk.)
"""
import os
import pickle
from functools import lru_cache

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SAFE = "safe"


@lru_cache(maxsize=1)
def _load():
    # The model file ships with this repo. Never load a pickle you didn't
    # create: unpickling can execute arbitrary code.
    with open(MODEL_PATH, "rb") as f:
        vectorizer, model = pickle.load(f)
    return vectorizer, model


def ml_detect(prompt: str) -> dict:
    vectorizer, model = _load()
    probs = model.predict_proba(vectorizer.transform([prompt]))[0]
    probabilities = {str(cls): float(p) for cls, p in zip(model.classes_, probs)}
    label = max(probabilities, key=probabilities.get)

    return {
        "label": label,
        "confidence": probabilities[label],
        "risk": 1.0 - probabilities.get(SAFE, 0.0),
        "probabilities": probabilities,
    }
