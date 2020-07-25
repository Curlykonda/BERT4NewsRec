#!/bin/bash
#SBATCH --job-name=get_logs
#SBATCH -n 4
#SBATCH -t 00:05:00
#SBATCH --mem=60000M


module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

#srun -n 2 -t 00:30:00 --pty bash -il

python -u source/evaluation/get_logs.py --local=0 --specified_only=1 --target_dir=logs_spec



