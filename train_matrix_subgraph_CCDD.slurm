#!/bin/bash
#SBATCH --job-name=rotate_matrix_subgraph_CCDD
#SBATCH --output=logs/rotate_matrix_%j.out
#SBATCH --error=logs/rotate_matrix_%j.err
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=shilpa14@ad.unc.edu
#SBATCH --mail-type=END

# Load modules
module purge
module load python/3.8.8
module load cuda/12.3
module load cudnn/8.9.7.29

# Activate environment
source /work/users/s/s/shilpa14/RotatE/rotate_env/bin/activate

# Navigate to project directory
cd /proj/jchunglab/projects/ec_moa/RotatE

# Check CUDA/GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# Run training
python -u codes/run.py --do_train \
  --cuda \
  --do_valid \
  --do_test \
  --data_path data/matrix_subgraph \
  --model RotatE \
  -n 128 -b 50 -d 500 \
  -g 24.0 -a 1.0 -adv \
  -lr 0.0001 \
  --max_steps 150000 \
  -save models/RotatE_MATRIX \
  --test_batch_size 4 \
  -de
