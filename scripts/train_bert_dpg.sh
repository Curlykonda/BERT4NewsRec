#!/bin/bash
#SBATCH --job-name=bert4rec_m_common
#SBATCH -n 8
#SBATCH -t 06:00:00
#SBATCH -p gpu_shared
#SBATCH --mem=60000M


module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

python --version

#srun -n 2 -t 00:30:00 --pty bash -il

data=("../Data/DPG_nov19/medium_time_split_most_common/")
embeddings="../embeddings/cc.nl.300.bin"
SEEDS=(42 113)


echo "$datapath"
for SEED in "${SEEDS[@]}"
do
  echo "$SEED"
  python -u main.py --template train_bert_dpg --dataset_code="DPG_nov19" --split="leave_one_out" --model_init_seed=$SEED
done


