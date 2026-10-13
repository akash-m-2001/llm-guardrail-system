"""Download public datasets and build training/data/train.csv and eval.csv.

Labels: safe | prompt_injection | harmful_content
eval.csv = 400 held-out prompts: 200 safe (50 of them "hard negatives"), 100 injection, 100 harmful.
"""
from pathlib import Path

import pandas as pd
from datasets import load_dataset

SEED = 42
OUT = Path("training/data")


def clean(text) -> str:
    return " ".join(str(text).split())  # collapse whitespace and newlines


rows = []

# 1. Prompt injections + normal prompts (Apache-2.0). label 1 = injection, 0 = normal.
injections = load_dataset("deepset/prompt-injections")
for split in ("train", "test"):
    for r in injections[split]:
        rows.append((clean(r["text"]), "prompt_injection" if r["label"] == 1 else "safe", "deepset"))

# 2. Harmful requests and their topic-matched harmless twins (MIT).
jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
rows += [(clean(r["Goal"]), "harmful_content", "jbb_harmful") for r in jbb["harmful"]]
rows += [(clean(r["Goal"]), "safe", "jbb_benign") for r in jbb["benign"]]

# 3. More harmful requests (CC BY-NC 4.0): prompts whose answers were judged unsafe.
beaver = load_dataset("PKU-Alignment/BeaverTails", split="30k_train").to_pandas()
harmful = beaver.loc[~beaver["is_safe"], "prompt"].map(clean).drop_duplicates()
rows += [(p, "harmful_content", "beavertails") for p in harmful.sample(1000, random_state=SEED)]

# 4. Normal instructions (CC BY-SA 3.0).
dolly = load_dataset("databricks/databricks-dolly-15k", split="train").to_pandas()
normal = dolly["instruction"].map(clean).drop_duplicates()
rows += [(p, "safe", "dolly") for p in normal.sample(1500, random_state=SEED)]

df = pd.DataFrame(rows, columns=["text", "label", "source"])
df = df[df["text"].str.len() > 0].drop_duplicates("text").reset_index(drop=True)


def take(mask, n):
    return df[mask].sample(n, random_state=SEED)


# Held-out evaluation set, chosen BEFORE training so the model never sees it.
eval_df = pd.concat([
    take(df.source == "jbb_benign", 50),                            # hard negatives: harmless but scary-sounding
    take((df.label == "safe") & (df.source != "jbb_benign"), 150),  # ordinary prompts
    take(df.label == "prompt_injection", 100),
    take(df.source == "jbb_harmful", 50),
    take(df.source == "beavertails", 50),
])
train_df = df.drop(eval_df.index)

OUT.mkdir(parents=True, exist_ok=True)
train_df.to_csv(OUT / "train.csv", index=False)
eval_df.to_csv(OUT / "eval.csv", index=False)

print("train:", len(train_df), "\n", train_df.label.value_counts().to_string())
print("eval: ", len(eval_df), "\n", eval_df.groupby(["label", "source"]).size().to_string())
