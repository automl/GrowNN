#!/bin/bash

#SBATCH --job-name=optimize_width
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=72:00:00

# Navigate to the project directory
cd /mnt/home/lfehring/MasterThesis/architectures-in-rl
module load Miniconda3
conda activate rl-architecture
export PYTHONPATH="$PYTHONPATH:/mnt/home/lfehring/MasterThesis/architectures-in-rl"
# Run the Python script with specified module

# Run the Python script with specified module
python baselines/blackbox_joined_hpo_nn_optimize_width/width_blackbox_joined_hpo_nn.py -m 
