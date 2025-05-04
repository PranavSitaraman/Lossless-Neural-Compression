#! /bin/bash
#SBATCH --requeue
#SBATCH --job-name=train
#SBATCH --output=train.out
#SBATCH --error=train.err

#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0

#SBATCH --time=01:00:00
#SBATCH --partition=gpu

module load python cuda cudnn
source .venv/bin/activate

export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
export MASTER_PORT=39591
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

srun --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
     python -u train.py
