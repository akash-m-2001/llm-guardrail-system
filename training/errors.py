"""Print the held-out prompts the full guardrail got wrong, grouped by source.

misses       = attacks it allowed       (false negatives)
false alarms = safe prompts it flagged  (false positives)
"""
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

eval_df = pd.read_csv("training/data/eval.csv")
client = TestClient(app)

rows = []
for i, r in eval_df.iterrows():
    response = client.post("/guardrail", json={"session_id": f"err-{i}", "prompt": r.text}).json()
    rows.append({
        "label": r.label,
        "source": r.source,
        "flagged": response["action"] != "allow",
        "ml_risk": response["metadata"]["ml_risk"],
        "ml_label": response["metadata"]["ml_label"],
        "text": r.text,
    })
res = pd.DataFrame(rows)

misses = res[(res.label != "safe") & ~res.flagged].sort_values("ml_risk")
false_alarms = res[(res.label == "safe") & res.flagged].sort_values("ml_risk", ascending=False)


def show(title, df):
    print(f"\n{'=' * 100}\n{title}: {len(df)}\n{'=' * 100}")
    for source, group in df.groupby("source"):
        print(f"\n--- {source} ({len(group)}) ---")
        for _, r in group.iterrows():
            print(f"  risk={r.ml_risk:.2f} model={r.ml_label:<16} {r.text[:110]}")


show("MISSES (attacks allowed)", misses)
show("FALSE ALARMS (safe prompts flagged)", false_alarms)