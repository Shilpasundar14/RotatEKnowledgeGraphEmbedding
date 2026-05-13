import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
input_file = Path("/proj/jchunglab/projects/ec_moa/KGs/ROBOKOP_30fd_baseline2_CCGGDD_noSubclassOf/raw/raw/rotorobo.txt")
output_dir = Path("data/rotate_protorobo_CCGGDD")
output_dir.mkdir(parents=True, exist_ok=True)

# Load filtered triples
df = pd.read_csv(input_file, sep='\t', header=None, names=["subject", "predicate", "object"])
df = df.dropna()

# Split: 80/10/10
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save splits
train_df.to_csv(output_dir / "train.txt", sep='\t', index=False, header=False)
valid_df.to_csv(output_dir / "valid.txt", sep='\t', index=False, header=False)
test_df.to_csv(output_dir / "test.txt", sep='\t', index=False, header=False)

# Create dicts
entities = pd.unique(df[["subject", "object"]].values.ravel("K"))
relations = pd.unique(df["predicate"])

with open(output_dir / "entities.dict", "w") as f:
    for idx, entity in enumerate(sorted(entities)):
        f.write(f"{idx}\t{entity}\n")

with open(output_dir / "relations.dict", "w") as f:
    for idx, rel in enumerate(sorted(relations)):
        f.write(f"{idx}\t{rel}\n")

print(f"Files saved to: {output_dir}")

