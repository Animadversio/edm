#!/bin/bash
#SBATCH -t 1-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner          # Partition to submit to
#SBATCH -c 16               # Number of cores (-c)
#SBATCH --mem=48G           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:4
#SBATCH --array 1-2
#SBATCH -o edm_train_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e edm_trainErr_%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=binxu_wang@hms.harvard.edu

echo "$SLURM_ARRAY_TASK_ID"
param_list=\
'--data=/n/home12/binxuwang/Github/edm/datasets/afhqv2-64x64-spectral-whiten.pt  --cond=0 --arch=ddpmpp --batch=256 --cres=1,2,2,2 --lr=2e-4 --dropout=0.05 --augment=0.15
--data=/n/home12/binxuwang/Github/edm/datasets/ffhq-64x64-spectral-whiten.pt  --cond=0 --arch=ddpmpp --batch=256 --cres=1,2,2,2 --lr=2e-4 --dropout=0.05 --augment=0.15
'

export param_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$param_name"


# load modules
module load python/3.10.9-fasrc01
module load cuda cudnn
mamba activate torch

# run code
cd ~/Github/edm
torchrun --standalone --nproc_per_node=4 train_tensor.py --outdir=training-runs  $param_name