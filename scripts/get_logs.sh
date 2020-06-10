#!/bin/bash
#SBATCH --job-name=get_logs
#SBATCH -n 2
#SBATCH -t 01:00:00
#SBATCH -p gpu_shared
#SBATCH --mem=60000M


module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

#srun -n 2 -t 00:30:00 --pty bash -il

python -u source/evaluation/get_logs.py --local=0



