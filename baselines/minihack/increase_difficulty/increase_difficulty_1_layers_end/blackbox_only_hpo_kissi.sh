#!/bin/bash

#SBATCH --job-name=baseline_increase_difficulty
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=72:00:00

# Navigate to the project directory
cd /mnt/home/lfehring/MasterThesis/architectures-in-rl
module load Miniconda3
conda activate rl-architecture
export PYTHONPATH="$PYTHONPATH:/mnt/home/lfehring/MasterThesis/architectures-in-rl"
# Run the Python script with specified module
python baselines/minihack/increase_difficulty/increase_difficulty_1_layers_end/blackbox_only_hpo.py -m
