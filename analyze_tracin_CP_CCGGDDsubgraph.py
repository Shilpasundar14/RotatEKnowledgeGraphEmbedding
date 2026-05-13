import pandas as pd

# Load your results file
df = pd.read_csv("tracin_cp_scores_CCGGDD_models2_10_26.csv")

print("Total rows:", len(df))
print("Example rows:\n", df.head())

grouped = df.groupby(['test_h', 'test_r', 'test_t'])
diversity_scores = grouped['train_r'].nunique().reset_index()
diversity_scores.columns = ['test_h', 'test_r', 'test_t', 'unique_train_relations']

# Sort to find the most diverse ones
diverse_tests = diversity_scores.sort_values(by='unique_train_relations', ascending=False)
print(diverse_tests.head(10))

# Pick top candidate
chosen_test = diverse_tests.iloc[1]

# Filter for this test triple
filtered = df[
    (df["test_h"] == chosen_test["test_h"]) &
    (df["test_r"] == chosen_test["test_r"]) &
    (df["test_t"] == chosen_test["test_t"])
]

# Sort by influence descending
filtered_sorted = filtered.sort_values(by="influence", ascending=False)

# Remove self-to-self
#filtered_sorted = filtered_sorted[
   #~(
        #(filtered_sorted["train_h"] == chosen_test["test_h"]) &
        #(filtered_sorted["train_r"] == chosen_test["test_r"]) &
        #(filtered_sorted["train_t"] == chosen_test["test_t"])
   # )
#]

# NEW: Remove zero influence scores
filtered_sorted = filtered_sorted[filtered_sorted["influence"] != 0.0]

# Show top 20 remaining
print(filtered_sorted.head(20))
# touch

# === Find self-to-self score for a specific test triple ===
target_h = "CHEBI:5781"
target_r = "predicate:16"
target_t = "HP:0002239"

self_score_row = df[
    (df["test_h"] == target_h) & 
    (df["test_r"] == target_r) & 
    (df["test_t"] == target_t) &
    (df["train_h"] == target_h) &
    (df["train_r"] == target_r) &
    (df["train_t"] == target_t)
]

if not self_score_row.empty:
    print("\n✅ Self-to-self TracIn score:")
    print(self_score_row[["influence", "checkpoint"]])
else:
    print("\n⚠️ No self-to-self row found for the selected test triple.")

print("\n Bottom 5 training triples with negative influence:")
bottom5 = filtered_sorted[filtered_sorted["influence"] < 0].sort_values(by="influence").head(5)
print(bottom5[["train_h", "train_r", "train_t", "influence", "checkpoint"]])

print("Total with influence < 0:", (filtered_sorted["influence"] < 0).sum())
print("Total with influence == 0:", (filtered_sorted["influence"] == 0).sum())
print("Total with influence > 0:", (filtered_sorted["influence"] > 0).sum())
