#!/bin/bash

#SBATCH --job-name=###
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --output=ssh_files/result_on_###.out
#SBATCH --error=ssh_files/error_on_###.err
#SBATCH --time=168:00:00
#SBATCH --nodelist=###
#SBATCH --partition=ai

# Navigate to the project directory
cd /bigwork/###/architectures-in-rl

# Run the Python script with specified module
/bigwork/###/.conda/envs/rl-architectures/bin/python ssh_files/open_sshtunnel.py
