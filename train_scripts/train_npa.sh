#!/bin/bash
#SBATCH --job-name=npa_vanilla
#SBATCH -n 8
#SBATCH -t 20:00:00
#SBATCH -p gpu_shared
#SBATCH --mem=60000M


module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

python --version

#srun -n 2 -t 00:30:00 --pty bash -il

#data=("../Data/DPG_nov19/medium_time_split_most_common/")
w_emb="./pc_word_embeddings/cc.nl.300.bin"
#pt_news_enc="./BertModelsPT/bert-base-dutch-cased"
art_len=(30)
SEEDS=(113 42)

d_art=400

lr=0.001
#decay_step=25
batch=128

exp_descr="npa_vanilla"

echo "$datapath"
for SEED in "${SEEDS[@]}"
do
  echo "$SEED"
  for LEN in "${art_len[@]}"
  do
    #1
  python -u main.py --template train_npa --model_init_seed=$SEED \
  --dim_art_emb $d_art  --pt_word_emb_path=$w_emb --lower_case=1 \
  --num_epochs=50 --max_article_len=$LEN \
  --lr $lr --cuda_launch_blocking=1 --train_batch_size=$batch --device="cuda" \
  --experiment_description $exp_descr l$LEN s$SEED
  done
done

