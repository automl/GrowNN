#!/bin/bash

#SBATCH --job-name=rl_training
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=72:00:00

# Navigate to the project directory
cd /mnt/home/lfehring/MasterThesis/architectures-in-rl
module load Miniconda3
conda activate rl-architecture
export PYTHONPATH="$PYTHONPATH:/mnt/home/lfehring/MasterThesis/architectures-in-rl"
# Run the Python script with specified module
python approach/bipedal_walker copy/net2deeper/ant/ant_n2d.py -m
