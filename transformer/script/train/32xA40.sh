#!/usr/bin/bash
#SBATCH -J transformer
#SBATCH --nodes=8
#SBATCH --gpus-per-node=A40:4
#SBATCH -t 2:00:00
#SBATCH --switches=1
#SBATCH -o log/%A/log.out
#SBATCH -e log/%A/err.out

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

mkdir -p log/$SLURM_JOB_ID
cp $2 log/$SLURM_JOB_ID/data_cfg.toml
cp $3 log/$SLURM_JOB_ID/train_cfg.toml

srun $1 \
    --nnodes=8 \
    --nproc_per_node=4 \
    --rdzv-backend=c10d \
    --rdzv-id=$RANDOM \
    --rdzv-endpoint=$head_node_ip:28052 \
    src/train.py --data-cfg log/$SLURM_JOB_ID/data_cfg.toml --train-cfg log/$SLURM_JOB_ID/train_cfg.toml

