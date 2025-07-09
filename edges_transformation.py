import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Set paths
edge_path = Path("/proj/jchunglab/projects/ec_moa/KGs/ROBOKOP_30fd_baseline2_CCDD/robokop_30fd_baseline2_CCDD_edges.tsv")
output_dir = Path("data/matrix_subgraph")
output_dir.mkdir(parents=True, exist_ok=True)

# Load and filter the data
df = pd.read_csv(edge_path, sep='\t', usecols=['subject', 'predicate', 'object'])
df = df.dropna(subset=['subject', 'predicate', 'object'])

# Train/Valid/Test split (80/10/10)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save triples
train_df.to_csv(output_dir / "train.txt", sep='\t', header=False, index=False)
valid_df.to_csv(output_dir / "valid.txt", sep='\t', header=False, index=False)
test_df.to_csv(output_dir / "test.txt", sep='\t', header=False, index=False)

# Build dictionaries from all triples
all_entities = pd.concat([
    train_df[['subject', 'object']],
    valid_df[['subject', 'object']],
    test_df[['subject', 'object']]
]).dropna().values.flatten()

entities = sorted(set(map(str, all_entities)))

relations = sorted(set(pd.concat([
    train_df['predicate'], valid_df['predicate'], test_df['predicate']
])))

entity2id = {entity: idx for idx, entity in enumerate(entities)}
relation2id = {rel: idx for idx, rel in enumerate(relations)}

# Save entities.dict
with open(output_dir / "entities.dict", 'w') as f:
    for entity, idx in entity2id.items():
        f.write(f"{idx}\t{entity}\n")

# Save relations.dict
with open(output_dir / "relations.dict", 'w') as f:
    for rel, idx in relation2id.items():
        f.write(f"{idx}\t{rel}\n")

print(f"Split completed. Files saved to {output_dir}")
