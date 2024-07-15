#!/bin/bash

#SBATCH --job-name=rl_training
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=24:00:00

# Navigate to the project directory
cd /mnt/home/lfehring/MasterThesis/architectures-in-rl
module load Miniconda3
conda activate rl-architecture
export PYTHONPATH="$PYTHONPATH:/mnt/home/lfehring/MasterThesis/architectures-in-rl"
# Run the Python script with specified module
python baselines/blackbox_warmstart_hpo/blackbox_warmstart_hpo.py -m
