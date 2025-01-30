#!/bin/bash

#SBATCH --job-name=4-layers-bb
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=72:00:00
#SBATCH --output=%x.out
#SBATCH --error=%x.err

# Navigate to the project directory
cd /mnt/home/l###/MasterThesis/architectures-in-rl
module load Miniconda3
conda activate rl-architecture
export PYTHONPATH="$PYTHONPATH:/mnt/home/l###/MasterThesis/architectures-in-rl"
# Run the Python script with specified module
python baselines/minihack/blackbox_only_hpo_4_layers/4_layers_bb.py -m
