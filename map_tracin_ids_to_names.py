import pandas as pd

# === File paths ===
entities_file = "/proj/jchunglab/projects/ec_moa/RotatE/data/rotate_protorobo_CCGGDD/entities.dict"
relations_file = "/proj/jchunglab/projects/ec_moa/RotatE/data/rotate_protorobo_CCGGDD/relations.dict"
input_csv = "/proj/jchunglab/projects/ec_moa/RotatE/tracin_cp_scores_CCGGDD_models2_10_26.csv"
output_csv = "/proj/jchunglab/projects/ec_moa/RotatE/tracin_cp_scores_CCGGDD_models2_10_26_mapped.csv"

# === Load entity and relation ID-to-name maps ===
entity_id_to_name = {}
with open(entities_file) as f:
    for line in f:
        idx, name = line.strip().split("\t")
        entity_id_to_name[idx] = name

relation_id_to_name = {}
with open(relations_file) as f:
    for line in f:
        idx, name = line.strip().split("\t")
        relation_id_to_name[idx] = name

# === Load the TracIn CSV file ===
df = pd.read_csv(input_csv)

# === Map function ===
def map_node(val):
    return entity_id_to_name.get(val.split(":")[-1], val)

def map_relation(val):
    return relation_id_to_name.get(val.split(":")[-1], val)

# === Apply mapping ===
df["train_h"] = df["train_h"].apply(map_node)
df["train_r"] = df["train_r"].apply(map_relation)
df["train_t"] = df["train_t"].apply(map_node)

df["test_h"] = df["test_h"].apply(map_node)
df["test_r"] = df["test_r"].apply(map_relation)
df["test_t"] = df["test_t"].apply(map_node)

# === Save the mapped CSV ===
df.to_csv(output_csv, index=False)
print(f"\n Mapped output saved to: {output_csv}")
