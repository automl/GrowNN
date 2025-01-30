#!/bin/bash

#SBATCH --job-name=increase_difficulty_two
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=72:00:00

# Navigate to the project directory
cd /mnt/home/l###/MasterThesis/architectures-in-rl
module load Miniconda3
conda activate rl-architecture
export PYTHONPATH="$PYTHONPATH:/mnt/home/l###/MasterThesis/architectures-in-rl"
# Run the Python script with specified module

# Run the Python script with specified module
python approach/minihack/increase_difficulty/net2deeper_depth_2/increase_difficulty_two.py -m 
