#! /bin/bash
#SBATCH --requeue
#SBATCH --job-name=train
#SBATCH --output=train.out
#SBATCH --error=train.err

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

#SBATCH --time=02:00:00
#SBATCH --mem=0
#SBATCH --partition=kempner
#SBATCH --constraint=a100
#SBATCH --account=kempner_sham_lab

module load python cuda cudnn
source .venv/bin/activate

export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
export MASTER_PORT=39591
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

srun --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
     python -u train.py