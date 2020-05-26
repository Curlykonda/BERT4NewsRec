#!/bin/bash
#SBATCH --job-name=bertje_embs
#SBATCH -n 4
#SBATCH -t 06:00:00
#SBATCH -p gpu_shared
#SBATCH --mem=60000M


module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling
export PYTHONIOENCODING=utf8

python --version

#srun -n 2 -t 00:30:00 --pty bash -il

data=("./Data/DPG_nov19/medium_time_split_n_rnd_users/news_data.pkl")
pt_model="./BertModelsPT/bert-base-dutch-cased"
max_len=(128 256)
lower_case=0

#embeddings="../embeddings/cc.nl.300.bin"
#SEEDS=(42 113)

echo "$data"
for len in "${max_len[@]}"
do
  echo "$len"

python -u source/preprocessing/compute_article_embs.py --data_dir $data --model_path $pt_model \
  --max_article_len $len --lower_case $lower_case

done


