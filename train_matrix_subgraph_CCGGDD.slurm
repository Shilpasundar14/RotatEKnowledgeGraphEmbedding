#!/bin/bash
#SBATCH --job-name=rotate_CCGGDD_tracin
#SBATCH --output=logs/rotate_CCGGDD_%j.out
#SBATCH --error=logs/rotate_CCGGDD_%j.err
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=shilpa14@ad.unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Load modules
module purge
module load python/3.12.1
module load cuda/12.9
module load cudnn/9.2.0.82

# Activate environment
source rotate_env_py39_v3/bin/activate

# Change to project directory
cd /proj/jchunglab/projects/ec_moa/RotatE

# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Slurm environment fixes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === Train on CCGGDD graph ===
python -u codes/run.py --do_train --cuda --do_valid --do_test --cpu_test \
  --data_path data/rotate_protorobo_CCGGDD \
  --model RotatE \
  -n 128 -b 128 -d 500 \
  -g 24.0 -a 1.0 -adv \
  -lr 0.0001 \
  --max_steps 728000 \
  --save_path models/RotatE_CCGGDD \
  --save_checkpoint_steps 36400 \
  --valid_steps 36400 \
  --test_batch_size 1 \
  -save models/RotatE_CCGGDD \
  -de
