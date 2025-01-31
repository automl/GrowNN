#!/bin/bash

#SBATCH --job-name=n2d_train
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=168:00:00
#SBATCH --nodelist=ai-n001
#SBATCH --partition=ai

# Navigate to the project directory
cd /bigwork/nhwpfehl/architectures-in-rl

# Run the Python script with specified module
/bigwork/nhwpfehl/.conda/envs/rl-architectures/bin/python approach/net2deeper/net2deeper.py -m

