#!/bin/bash

#SBATCH --job-name=new_targets_labelnoise1N1024_L7_r1_wd1e-05_epochs60100
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --output=log/new_targets_labelnoise1/N1024_L7_r1_wd1e-05_epochs60100.out
#SBATCH --error=log/new_targets_labelnoise1/N1024_L7_r1_wd1e-05_epochs60100.err
echo "$date Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
echo "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
which python

python -W error::UserWarning run_job.py --filename new_targets_labelnoise1 --datasetsize 1024 --L 7 --r 1 --labelnoise 1 --weight_decay 1e-05 --epochs 60100