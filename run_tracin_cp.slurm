#!/bin/bash
#SBATCH --job-name=compute_tracin_cp
#SBATCH --output=logs/tracin_cp_output_%j.out 
#SBATCH --error=logs/tracin_cp_error_%j.err 
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access  
#SBATCH --gres=gpu:1      
#SBATCH --time=08:00:00   
#SBATCH --mem=50G         
#SBATCH --cpus-per-task=4  
#SBATCH --mail-user=shilpa14@ad.unc.edu  
#SBATCH --mail-type=ALL    

# Load required modules
module purge
module load python/3.9.6   
module load cuda/12.3
module load cudnn/8.9.7.29

# Activate your virtual environment
source /proj/jchunglab/projects/ec_moa/RotatE/rotate_env_py39_v3/bin/activate

# Navigate to your project directory (adjust path if needed)
cd /proj/jchunglab/projects/ec_moa/RotatE

# Verify GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# Run the TracInCP computation script
python -u compute_tracin_cp.py  
