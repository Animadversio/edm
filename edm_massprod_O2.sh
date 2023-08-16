#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu_quad
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --array=25-26
#SBATCH --mail-user=binxu_wang@hms.harvard.edu
#SBATCH -o ffhq64_hyrbid_samp_%j.%a.out

echo "$SLURM_ARRAY_TASK_ID"
#--skipstep-list $SKIPSTEP_LIST
param_list=\
'--seeds_range    0   4096   --max-batch-size 64  --dataset-name ffhq64
--seeds_range  4096   8192   --max-batch-size 64  --dataset-name ffhq64
--seeds_range  8192  12288   --max-batch-size 64  --dataset-name ffhq64
--seeds_range 12288  16384   --max-batch-size 64  --dataset-name ffhq64
--seeds_range 16384  20480   --max-batch-size 64  --dataset-name ffhq64
--seeds_range 20480  24576   --max-batch-size 64  --dataset-name ffhq64
--seeds_range     0   4096   --max-batch-size 64  --dataset-name afhq64
--seeds_range  4096   8192   --max-batch-size 64  --dataset-name afhq64
--seeds_range  8192  12288   --max-batch-size 64  --dataset-name afhq64
--seeds_range 12288  16384   --max-batch-size 64  --dataset-name afhq64
--seeds_range 16384  20480   --max-batch-size 64  --dataset-name afhq64
--seeds_range 20480  24576   --max-batch-size 64  --dataset-name afhq64
--seeds_range 24576  28672   --max-batch-size 64  --dataset-name ffhq64
--seeds_range 28672  32768   --max-batch-size 64  --dataset-name ffhq64
--seeds_range 32768  36864   --max-batch-size 64  --dataset-name ffhq64
--seeds_range 36864  40960   --max-batch-size 64  --dataset-name ffhq64
--seeds_range 40960  45056   --max-batch-size 64  --dataset-name ffhq64
--seeds_range 45056  49152   --max-batch-size 64  --dataset-name ffhq64
--seeds_range 24576  28672   --max-batch-size 64  --dataset-name afhq64
--seeds_range 28672  32768   --max-batch-size 64  --dataset-name afhq64
--seeds_range 32768  36864   --max-batch-size 64  --dataset-name afhq64
--seeds_range 36864  40960   --max-batch-size 64  --dataset-name afhq64
--seeds_range 40960  45056   --max-batch-size 64  --dataset-name afhq64
--seeds_range 45056  49152   --max-batch-size 64  --dataset-name afhq64
--seeds_range 49152  50048   --max-batch-size 64  --dataset-name ffhq64
--seeds_range 49152  50048   --max-batch-size 64  --dataset-name afhq64
'

export unit_name="$(echo "$param_list" | head -n $SLURM_ARRAY_TASK_ID | tail -1)"
echo "$unit_name"

module load gcc/6.2.0
module load cuda/10.2
#module load conda2/4.2.13

#conda init bash
source  activate torch

cd ~/Github/edm
python3 edm_analytical_massprod_O2.py  $unit_name