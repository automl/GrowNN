#!/bin/bash

#SBATCH --job-name=ant_n2d_2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=72:00:00
#SBATCH --output=%x.out
#SBATCH --error=%x.err

# Navigate to the project directory
cd /mnt/home/lfehring/MasterThesis/architectures-in-rl
module load Miniconda3
conda activate rl-architecture
export PYTHONPATH="$PYTHONPATH:/mnt/home/lfehring/MasterThesis/architectures-in-rl"
# Run the Python script with specified module
python approach/ant/net2deeper/budget2/ant_n2d.py -m
