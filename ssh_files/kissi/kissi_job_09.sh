#!/bin/bash

#SBATCH --job-name=ssh_gpu9.kisski
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --output=ssh_files/kissi/result_on_gpu9.kisski.out
#SBATCH --error=ssh_files/kissi/error_on_gpu9.kisski.err
#SBATCH --time=168:00:00
#SBATCH --nodelist=gpu009.kisski

cd /mnt/home/lfehring/MasterThesis/architectures-in-rl
module load Miniconda3
conda activate rl-architecture
export PYTHONPATH="$PYTHONPATH:/mnt/home/lfehring/MasterThesis/architectures-in-rl"

# Run the Python script with specified module
python ssh_files/open_sshtunnel.py
