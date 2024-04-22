#!/bin/bash

#SBATCH --job-name=ai-n001
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --output=ssh_files/result_on_ai-n001.out
#SBATCH --error=ssh_files/error_on_ai-n001.err
#SBATCH --time=48:00:00
#SBATCH --nodelist=ai-n001
#SBATCH --partition=ai

# Navigate to the project directory
cd /bigwork/nhwpfehl/architectures-in-rl

# Run the Python script with specified module
/bigwork/nhwpfehl/.conda/envs/rl-architectures/bin/python ssh_files/open_sshtunnel.py
