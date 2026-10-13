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

# 3. More harmful requests (CC BY-NC 4.0). BeaverTails labels the *answers*, not the
#    questions: a harmless question can still get an unsafe answer. Only keep questions
#    where EVERY answer (at least 2) was unsafe, so the question itself is the problem.
beaver = load_dataset("PKU-Alignment/BeaverTails", split="30k_train").to_pandas()
beaver["prompt"] = beaver["prompt"].map(clean)
answers = beaver.groupby("prompt")["is_safe"].agg(["count", "sum"])
harmful = answers[(answers["sum"] == 0) & (answers["count"] >= 2)].index.to_series()
rows += [(p, "harmful_content", "beavertails") for p in harmful.sample(1000, random_state=SEED)]
# 4. Normal instructions (CC BY-SA 3.0).
dolly = load_dataset("databricks/databricks-dolly-15k", split="train").to_pandas()
normal = dolly["instruction"].map(clean).drop_duplicates()
rows += [(p, "safe", "dolly") for p in normal.sample(1500, random_state=SEED)]

# 5. XSTest (CC-BY-4.0): harmless questions that SOUND dangerous ("How can I kill a
#    Python process?"), plus genuinely unsafe ones. Breaks the shortcut "question = harmful".
xstest = pd.read_csv("https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv")
rows += [(clean(p), "safe", "xstest_safe") for p in xstest.loc[xstest.label == "safe", "prompt"]]
rows += [(clean(p), "harmful_content", "xstest_unsafe") for p in xstest.loc[xstest.label == "unsafe", "prompt"]]

# 6. HarmBench (MIT): harmful requests phrased as COMMANDS ("Give detailed instructions
#    for..."). Breaks the shortcut "command = safe". Standard behaviors only.
harmbench = pd.read_csv(
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
rows += [(clean(p), "harmful_content", "harmbench") for p in harmbench.loc[harmbench.FunctionalCategory == "standard", "Behavior"]]
df = pd.DataFrame(rows, columns=["text", "label", "source"])
df = df[df["text"].str.len() > 0].drop_duplicates("text").reset_index(drop=True)


def take(mask, n):
    return df[mask].sample(n, random_state=SEED)


# Held-out evaluation set, chosen BEFORE training so the model never sees it.
eval_df = pd.concat([
    take(df.source == "jbb_benign", 50),                            # hard negatives: harmless but scary-sounding
    take(df.source.isin(["deepset", "dolly"]) & (df.label == "safe"), 150),  # ordinary prompts
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
