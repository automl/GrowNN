#!/bin/bash

# Loop through the job scripts from kissi_job01.sh to kissi_job09.sh
for i in {01..09}
do
  # Construct the job script name
  job_script="ssh_files/kissi/kissi_job_${i}.sh"
  
  # Check if the job script exists before submitting
  if [ -f "$job_script" ]; then
    echo "Submitting job: $job_script"
    sbatch "$job_script"
  else
    echo "Job script $job_script does not exist. Skipping."
  fi
done
