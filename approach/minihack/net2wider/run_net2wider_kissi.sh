#!/bin/bash

#SBATCH --job-name=n2w_train
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
python approach/minihack/net2wider/net2wider.py -m non_hyperparameters.environment_name="MiniHack-Room-Monster-10x10-v0"
