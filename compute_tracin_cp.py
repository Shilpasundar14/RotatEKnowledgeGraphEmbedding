import torch
from model import KGEModel
import os
from tqdm import tqdm

# === CONFIG ===
checkpoint_dir = "/proj/jchunglab/projects/ec_moa/KGs/ROBOKOP_30fd_baseline2_CCDD_noSubclassOf/protoroborotor_derived/CCDD/trained_models"
checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")])

entities_file = "data/rotate_protorobo_CCDD/entities.dict"
relations_file = "data/rotate_protorobo_CCDD/relations.dict"
train_file = "data/rotate_protorobo_CCDD/train.txt"
test_file = "data/rotate_protorobo_CCDD/test.txt"

output_file = "tracin_cp_scores.csv"

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

# === Load a few test triples (pick first 5 for demo) ===
test_triples = []
with open(test_file) as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        h, r, t = line.strip().split("\t")
        test_triples.append((h, r, t))

# === Load a few train triples (pick first 100 for demo) ===
train_triples = []
with open(train_file) as f:
    for i, line in enumerate(f):
        if i >= 100:
            break
        h, r, t = line.strip().split("\t")
        train_triples.append((h, r, t))

# === Open output ===
with open(output_file, "w") as out:
    out.write("checkpoint,train_h,train_r,train_t,test_h,test_r,test_t,influence\n")

    for ckpt in tqdm(checkpoint_files):
        ckpt_path = os.path.join(checkpoint_dir, ckpt)

        # Load model
        model = KGEModel(
            model_name="RotatE",
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=500,
            gamma=24.0,
            double_entity_embedding=True,
            double_relation_embedding=False
        )

        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint)
        model.eval()

        for test in test_triples:
            test_tensor = torch.LongTensor([[entity2id[test[0]], relation2id[test[1]], entity2id[test[2]]]])
            test_tensor.requires_grad = True

            model.zero_grad()
            test_score = model(test_tensor)
            test_loss = torch.logsigmoid(test_score).mean()
            test_loss.backward()

            test_grads = []
            for p in model.parameters():
                if p.grad is not None:
                    test_grads.append(p.grad.detach().clone().view(-1))
            test_grad_flat = torch.cat(test_grads)

            for train in train_triples:
                train_tensor = torch.LongTensor([[entity2id[train[0]], relation2id[train[1]], entity2id[train[2]]]])
                train_tensor.requires_grad = True

                model.zero_grad()
                train_score = model(train_tensor)
                train_loss = torch.logsigmoid(train_score).mean()
                train_loss.backward()

                train_grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        train_grads.append(p.grad.detach().clone().view(-1))
                train_grad_flat = torch.cat(train_grads)

                # Dot product
                influence = torch.dot(train_grad_flat, test_grad_flat).item()

                # Write to file
                out.write(f"{ckpt},{train[0]},{train[1]},{train[2]},{test[0]},{test[1]},{test[2]},{influence}\n")

