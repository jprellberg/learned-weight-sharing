#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=6-00:00
#SBATCH --partition=long


echo "Starting $1 parallel processes on a single GPU"

for i in $(seq 1 $1); do
    PYTHONPATH="$PYTHONPATH:$(pwd)" python3.6 -u "$2" "$SLURM_ARRAY_TASK_ID" &
    pids[$i]=$!
done

for pid in ${pids[*]}; do
    wait $pid
done
