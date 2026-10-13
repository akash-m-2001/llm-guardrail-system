"""Train TF-IDF + logistic regression on train.csv and save it to app/core/model.pkl."""
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

train = pd.read_csv("training/data/train.csv")

from sklearn.pipeline import FeatureUnion

# Word features (meaning) + character features (robust to typos, leetspeak, other languages).
vectorizer = FeatureUnion([
    ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
    ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
])
model = LogisticRegression(class_weight="balanced", max_iter=2000, C=4.0)

# 5-fold cross-validation on the TRAINING data only, a sanity check before the real test.
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(make_pipeline(vectorizer, model), train.text, train.label, cv=cv, scoring="f1_macro")
print(f"5-fold macro F1 on training data: {scores.mean():.3f} (+/- {scores.std():.3f})")
model.fit(vectorizer.fit_transform(train.text), train.label)

# Same format the API already expects: a (vectorizer, model) tuple.
with open("app/core/model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)
print("saved app/core/model.pkl, classes:", list(model.classes_))