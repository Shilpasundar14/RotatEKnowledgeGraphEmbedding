import os
import pandas as pd

# Load your results file
df = pd.read_csv("tracin_cp_scores_CCGGDD_models2_10_26.csv")

test_only = df[
    (df["test_h"] == "CHEBI:5781") &
    (df["test_r"] == "predicate:16") &
    (df["test_t"] == "HP:0002239")
]

print(f"Test triple appears {len(test_only)} times in the CSV.")

train_only = df[
    (df["train_h"] == "CHEBI:5781") &
    (df["train_r"] == "predicate:16") &
    (df["train_t"] == "HP:0002239")
]

print(f"Training triple appears {len(train_only)} times in the CSV.")
df["train_h"] = df["train_h"].str.strip()
df["train_r"] = df["train_r"].str.strip()
df["train_t"] = df["train_t"].str.strip()
df["test_h"] = df["test_h"].str.strip()
df["test_r"] = df["test_r"].str.strip()
df["test_t"] = df["test_t"].str.strip()

self_score_row = df[
    (df["test_h"] == "CHEBI:5781") &
    (df["test_r"] == "predicate:16") &
    (df["test_t"] == "HP:0002239") &
    (df["train_h"] == "CHEBI:5781") &
    (df["train_r"] == "predicate:16") &
    (df["train_t"] == "HP:0002239")
]

print(self_score_row)
