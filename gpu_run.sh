#!/bin/sh -l
#SBATCH -A amannodi          # amannodi from amannodi-f
#SBATCH --qos=normal         # normal/standby
#SBATCH --partition=a100-40gb     # partition that matches your available GPUs
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH -t 3-00:00:00
#SBATCH --job-name=MACE-Scratch
#SBATCH --output=mace_training_%j.out
#SBATCH --error=mace_training_%j.err

# Load necessary modules (adjust for your system)
module load conda

# Activate your conda environment (if using conda)
conda activate mace_gpu

nvidia-smi

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"

# Run MACE training
echo "Starting MACE training..."
mace_run_train --config ./configs/scratch_config.yaml

# Check exit status
if [ $? -eq 0 ]; then
    echo "MACE training completed successfully!"
else
    echo "MACE training failed with exit code $?"
fi

echo "End Time: $(date)"