import torch
import sys
sys.path.append('/proj/jchunglab/projects/ec_moa/RotatE/codes')
from model import KGEModel
import os
from tqdm import tqdm
import torch.nn.functional as F

# === CONFIG ===
checkpoint_dir = "/proj/jchunglab/projects/ec_moa/KGs/ROBOKOP_30fd_baseline2_CCGGDD_noSubclassOf/trained_models"
checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")])

entities_file = "data/rotate_protorobo_CCGGDD/entities.dict"
relations_file = "data/rotate_protorobo_CCGGDD/relations.dict"
train_file = "data/rotate_protorobo_CCGGDD/train.txt"
test_file = "data/rotate_protorobo_CCGGDD/test_batch5.txt"

output_file = "tracin_cp_scores_CCGGDD_models2_10_26.csv"

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
print(f"Size of the entity: {nentity}, Size of the relation: {nrelation}")

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
file_exists = os.path.exists(output_file)
with open(output_file, "a") as out:
    if not file_exists:
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

        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False) # Load only the model parameters 
        model.eval()

        for test in test_triples:
            # Keep as LongTensor for indexing
            test_tensor = torch.LongTensor([[entity2id[test[0]], relation2id[test[1]], entity2id[test[2]]]])

            # Convert to FloatTensor for forward propagation
            float_test_tensor = test_tensor.float() 
            float_test_tensor.requires_grad = True

            model.zero_grad()
            test_score = model(float_test_tensor)
            test_loss = F.logsigmoid(test_score).mean()
            test_loss.backward()

            test_grads = []
            for p in model.parameters():
                if p.grad is not None:
                    test_grads.append(p.grad.detach().clone().view(-1))
            test_grad_flat = torch.cat(test_grads)  # Flatten gradients for the test set

            for train in train_triples:
                # Keep as LongTensor for indexing
                train_tensor = torch.LongTensor([[entity2id[train[0]], relation2id[train[1]], entity2id[train[2]]]])

                # Convert to FloatTensor for forward propagation
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

                # Dot product
                influence = torch.dot(train_grad_flat, test_grad_flat).item()

                # Write to csv file
                out.write(f"{ckpt},{train[0]},{train[1]},{train[2]},{test[0]},{test[1]},{test[2]},{influence}\n")

print(f"Influence scores saved to {output_file}")