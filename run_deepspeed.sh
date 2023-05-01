#!/usr/bin/env bash
#SBATCH --job-name=gist
#SBATCH --ntasks=1
#SBATCH --mem=480gb
#SBATCH --time=26:00:00
#SBATCH --output=gist.log
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4

# This script can either be used interactively or submitted to SLURM with
# sbatch.

# NOTE: LLaMA runs typically take ~7 hours; FLAN-T5-XXL runs typically take ~26
# hours. You can probably get away with training FLAN-T5-XXL less.


TAG="test"

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port $port --num_gpus=4 --no_local_rank \
    --module src.train \
    +model=llama-7b wandb.tag=$TAG \
    training.deepspeed=ds_configs/stage3.json \
    training.gist.condition=gist \
    training.gist.num_gist_tokens=1
