import torch
import sys
sys.path.append('/proj/jchunglab/projects/ec_moa/RotatE/codes')
from model import KGEModel
import os
import torch.nn.functional as F
from tqdm import tqdm
import time

# === CONFIG ===
checkpoint_path = "/proj/jchunglab/projects/ec_moa/KGs/ROBOKOP_30fd_baseline2_CCGGDD_noSubclassOf/trained_models/model_2.pt"

entities_file = "data/rotate_protorobo_CCGGDD/entities.dict"
relations_file = "data/rotate_protorobo_CCGGDD/relations.dict"
train_file = "data/rotate_protorobo_CCGGDD/train.txt"

# === Use one fixed test triple ===
test_h = "CHEBI:5781"
test_r = "predicate:16"
test_t = "HP:0002239"

# === Load dicts ===
entity2id = {}
relation2id = {}

with open(entities_file) as f:
    for line in f:
        idx, ent = line.strip().split("\t")
        entity2id[ent] = int(idx)

with open(relations_file) as f:
    for line in f:
        idx, rel = line.strip().split("\t")
        relation2id[rel] = int(idx)

nentity = len(entity2id)
nrelation = len(relation2id)

# === Prepare test triple ===
test_tensor = torch.LongTensor([[entity2id[test_h], relation2id[test_r], entity2id[test_t]]])
float_test_tensor = test_tensor.float()
float_test_tensor.requires_grad = True

# === Load Model ===
model = KGEModel(
    model_name="RotatE",
    nentity=nentity,
    nrelation=nrelation,
    hidden_dim=500,
    gamma=24.0,
    double_entity_embedding=True,
    double_relation_embedding=False
)

checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

# === Get gradient for test triple ===
model.zero_grad()
test_score = model(float_test_tensor)
test_loss = F.logsigmoid(test_score).mean()
test_loss.backward()

test_grads = []
for p in model.parameters():
    if p.grad is not None:
        test_grads.append(p.grad.detach().clone().view(-1))
test_grad_flat = torch.cat(test_grads)

# === Benchmark: Loop over training triples ===
start_time = time.time()

train_triples = []
with open(train_file) as f:
    for line in f:
        h, r, t = line.strip().split("\t")
        train_triples.append((h, r, t))

# Optional: To just test speed with HALF
# train_triples = train_triples[:4250000]

print(f"Total train triples loaded: {len(train_triples)}")

# Run through all and calculate dummy dot product
for train in tqdm(train_triples):
    train_tensor = torch.LongTensor([[entity2id[train[0]], relation2id[train[1]], entity2id[train[2]]]])
    float_train_tensor = train_tensor.float()
    float_train_tensor.requires_grad = True

    model.zero_grad()
    train_score = model(float_train_tensor)
    train_loss = F.logsigmoid(train_score).mean()
    train_loss.backward()

    train_grads = []
    for p in model.parameters():
        if p.grad is not None:
            train_grads.append(p.grad.detach().clone().view(-1))
    train_grad_flat = torch.cat(train_grads)

    # Compute dot product (TracIn score)
    influence = torch.dot(train_grad_flat, test_grad_flat).item()

end_time = time.time()

total_time = end_time - start_time
print(f"\n Total time for 1 test triple against {len(train_triples)} train triples: {total_time / 60:.2f} minutes")

# Project time for 10,000 test triples
projected_time_days = (total_time * 10000) / (60 * 60 * 24)
print(f"Projected time for 10,000 test triples: {projected_time_days:.2f} days")
