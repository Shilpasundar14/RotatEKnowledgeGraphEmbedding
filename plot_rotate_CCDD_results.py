import matplotlib.pyplot as plt
import re

# Your log file path
log_path = "/proj/jchunglab/projects/ec_moa/RotatE/rotate_CCDD.log"

# Containers
epochs = []
losses = []
val_epochs = []
val_mrrs = []
val_hits10 = []
val_mrs = []

# Read and parse
with open(log_path, "r") as f:
    for line in f:
        loss_match = re.match(r"Epoch:\s*(\d+),\s*Loss:\s*([\d.]+)", line)
        val_match = re.match(
            r"Epoch:\s*(\d+),\s*Val Mean Rank:\s*([\d.]+),\s*Val MRR:\s*([\d.]+),\s*Val Hits@10:\s*([\d.]+)",
            line,
        )
        if loss_match:
            epochs.append(int(loss_match.group(1)))
            losses.append(float(loss_match.group(2)))
        elif val_match:
            val_epochs.append(int(val_match.group(1)))
            val_mrs.append(float(val_match.group(2)))
            val_mrrs.append(float(val_match.group(3)))
            val_hits10.append(float(val_match.group(4)))

# Your preferred colors
color1 = "#4D4D4D"  # dark grey
color2 = "#008080"  # teal

# 1. Training Loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, color=color2)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("RotatE Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("rotate_CCDD_training_loss.png")

# 2. Training Loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, color=color2)
plt.xlabel("Epoch")
plt.ylabel("Training Loss (log scale)")
plt.title("RotatE Training Loss (log scale)")
plt.yscale("log")
plt.grid(True)
plt.tight_layout()
plt.savefig("rotate_CCDD_training_loss_logscale.png")

# 3. Validation MRR
plt.figure(figsize=(8, 5))
plt.plot(val_epochs, val_mrrs, color=color2)
plt.xlabel("Epoch")
plt.ylabel("Validation MRR")
plt.title("RotatE Validation MRR")
plt.grid(True)
plt.tight_layout()
plt.savefig("rotate_CCDD_val_mrr.png")

# 4. Validation Hits@10
plt.figure(figsize=(8, 5))
plt.plot(val_epochs, val_hits10, color=color1)
plt.xlabel("Epoch")
plt.ylabel("Validation Hits@10")
plt.title("RotatE Validation Hits@10")
plt.grid(True)
plt.tight_layout()
plt.savefig("rotate_CCDD_val_hits10.png")

# 5. Validation Mean Rank
plt.figure(figsize=(8, 5))
plt.plot(val_epochs, val_mrs, color=color1)
plt.xlabel("Epoch")
plt.ylabel("Validation Mean Rank")
plt.title("RotatE Validation Mean Rank")
plt.grid(True)
plt.tight_layout()
plt.savefig("rotate_CCDD_val_mr.png")

print("5 Plots saved successfully.")
