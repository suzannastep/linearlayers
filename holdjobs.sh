#!/bin/bash

#commands to hold/release/cancel hobs
# for i in $(seq 106428 106509); do scontrol hold $i; done
# for i in $(seq 106428 106509); do scontrol release $i; done
# for i in $(seq 109204 109470); do scancel $i; done

# squeue -u $USER | grep 197 | awk '{print $1}' | xargs -n 1 scancel

# useful for making sure jobs are running on GPU and aren't having any errors
# grep -irm 1 cuda  *.out | wc -l
# grep -irm 1 cpu  *.out | wc -l
# ls -1 *.out | wc -l
# cat *.err

#useful printout of queue with longer printout of job name
# squeue --format="%.10i %.9P %.75j %.8u %.8T %.10M %.9l %.6D %R" -u sueparkinson

#request GPU
# srun -p general --gres=gpu:1 --pty bash