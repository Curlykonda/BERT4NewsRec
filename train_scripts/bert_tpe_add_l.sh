#!/bin/bash
#SBATCH --job-name=bertje_pe_l
#SBATCH -n 8
#SBATCH -t 37:00:00
#SBATCH -p gpu_shared
#SBATCH --mem=60000M


module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

python --version

#srun -n 2 -t 00:30:00 --pty bash -il

data=("./Data/DPG_nov19/100k_time_split_n_rnd_users/")
#embeddings="../embeddings/cc.nl.300.bin"
pt_news_enc="BERTje"
pt_news_enc_path="./BertModelsPT/bert-base-dutch-cased"

SEED=$SLURM_ARRAY_TASK_ID

art_len=30
hist_len=50

POS_EMBS=("tpe")
neg_ratios=(4 9 24)

nie="lin_gelu"
lr=0.001

n_users=100000
COUNTER=0
#####

exp_descr="100k_add"

for K in "${neg_ratios[@]}"
do
  for POS in "${POS_EMBS[@]}"
  do
    echo "$exp_descr $POS al$art_len hl$hist_len k$K s$SEED"
      #1
    python -u main.py --template train_bert_pcp --model_init_seed=$SEED --dataset_path=$data \
    --train_negative_sampler_code random --train_negative_sample_size=$K \
    --pt_news_enc=$pt_news_enc --path_pt_news_enc=$pt_news_enc_path \
    --pos_embs=$POS --nie_layer $nie \
    --max_article_len=$art_len --max_hist_len=$hist_len \
    --lr $lr --n_users=$n_users --cuda_launch_blocking=1 \
    --experiment_description $exp_descr $POS al$art_len hl$hist_len k$K s$SEED

    ((COUNTER++))
    echo "Exp counter: $COUNTER"
  done
done