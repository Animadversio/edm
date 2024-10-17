#!/bin/bash
#SBATCH -t 3:00:00                      # Runtime in HH:MM:SS
#SBATCH -p kempner_h100                       # Partition to submit to
#SBATCH -N 1                             # Number of nodes
#SBATCH -n 16                            # Total number of cores
#SBATCH --mem=80G                       # Memory pool for all cores
#SBATCH --gres=gpu:1                     # Number of GPUs per node
#SBATCH --array=1-6                      # Array job with 5 tasks
#SBATCH -o dpm_fid_eval_%A_%a.out        # STDOUT file
#SBATCH -e dpm_fid_eval_%A_%a.err        # STDERR file
#SBATCH --mail-user=binxu_wang@hms.harvard.edu  # Email for notifications
#SBATCH --mail-type=END,FAIL             # When to send emails

echo "Running task ID: $SLURM_ARRAY_TASK_ID"

# Define parameters for each node/task
param_list=(
    "--dataset_name ffhq64 --range_start 0 --range_end 65"
    "--dataset_name ffhq64 --range_start 65 --range_end 130"
    "--dataset_name ffhq64 --range_start 130 --range_end 195"
    "--dataset_name ffhq64 --range_start 195 --range_end 260"
    "--dataset_name ffhq64 --range_start 260 --range_end 325"
    "--dataset_name ffhq64 --range_start 325 --range_end 390"
)

# Select the parameters based on the task ID
param_index=$((SLURM_ARRAY_TASK_ID - 1))
params=${param_list[$param_index]}
echo "Parameters: $params"
# Activate the Python environment
mamba activate torch2
# Navigate to the project directory
cd /n/home12/binxuwang/Github/edm
# Execute the evaluation script with the selected parameters
python DPM_sampler_analytical_fid_eval.py $params