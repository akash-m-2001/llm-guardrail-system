"""Pick the ML threshold using ONLY the training data (cross-validated), never the eval set.

For each threshold: what share of attacks would the ML model catch, and what
share of safe prompts would it wrongly flag?
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

train = pd.read_csv("training/data/train.csv")
from sklearn.pipeline import FeatureUnion

# Must match training/train.py exactly, or the threshold is chosen for a different model.
pipeline = make_pipeline(
    FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
    ]),
    LogisticRegression(class_weight="balanced", max_iter=2000, C=4.0),
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Each prompt is scored by a model that did NOT see it during training.
probs = cross_val_predict(pipeline, train.text, train.label, cv=cv, method="predict_proba")
classes = sorted(train.label.unique())  # same order sklearn uses
risk = 1 - probs[:, classes.index("safe")]
attack = (train.label != "safe").values
hard = (train.source == "jbb_benign").values

print(f"{'threshold':>9} {'detection':>10} {'false_pos':>10} {'fp_hard_neg':>12}")
for t in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
    flagged = risk > t
    print(f"{t:>9} {flagged[attack].mean():>10.3f} {flagged[~attack].mean():>10.3f} {flagged[hard].mean():>12.3f}")