"""Evaluate 4 configurations on the 400 held-out prompts.

detection rate      = share of attacks flagged      (higher is better)
false-positive rate = share of safe prompts flagged (lower is better)
"""
import json
import sys

import pandas as pd
from fastapi.testclient import TestClient

from app.core.detector import detect_risk
from app.core.ml_detector import ml_detect
from app.api.routes import ML_THRESHOLD
from app.core.sanitization import normalize
from app.main import app

results_file = sys.argv[1] if len(sys.argv) > 1 else "training/results.json"
eval_df = pd.read_csv("training/data/eval.csv")
client = TestClient(app)

rows = []
for i, r in eval_df.iterrows():
    text = normalize(r.text)
    rules, ml = detect_risk(text), ml_detect(text)
    # A fresh session per prompt, so one prompt's history can't affect another's result.
    response = client.post("/guardrail", json={"session_id": f"eval-{i}", "prompt": r.text}).json()
    rows.append({
        "label": r.label,
        "source": r.source,
        "rules_only": bool(rules["categories"]),
        "ml_only": ml["label"] != "safe",
        "hybrid": max(rules["prompt_risk_score"], ml["risk"] if ml["risk"] > ML_THRESHOLD else 0.0) > 0.5,
        "full_pipeline": response["action"] != "allow",
    })

res = pd.DataFrame(rows)
attack = res.label != "safe"
configs = ["rules_only", "ml_only", "hybrid", "full_pipeline"]
summary = {
    config: {
        "detection_rate": round(res.loc[attack, config].mean(), 3),
        "false_positive_rate": round(res.loc[~attack, config].mean(), 3),
        "fpr_hard_negatives": round(res.loc[res.source == "jbb_benign", config].mean(), 3),
        "injection_detection": round(res.loc[res.label == "prompt_injection", config].mean(), 3),
        "harmful_detection": round(res.loc[res.label == "harmful_content", config].mean(), 3),
    }
    for config in configs
}
print(pd.DataFrame(summary).T.to_string())

# Flag rate per source: reveals whether the model learned a dataset's style instead of intent.
print("\nFlag rate by source:")
print(res.groupby(["label", "source"])[configs].mean().round(3).to_string())

with open(results_file, "w") as f:
    json.dump(summary, f, indent=2)