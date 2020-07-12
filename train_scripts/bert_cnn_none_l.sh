#!/bin/bash
#SBATCH --job-name=bert_cnn_none
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
w_emb="./pc_word_embeddings/cc.nl.300.bin"
#pt_news_enc="./BertModelsPT/bert-base-dutch-cased"
SEED=$SLURM_ARRAY_TASK_ID

art_len=30


POS=None #
neg_ratios=(4 9 24)

enc="wucnn"
d_art=400

nie="lin_gelu"
lr=0.001

n_users=100000
exp_descr="100k_NpaCNN"
COUNTER=0

echo "$data"
echo "$exp_descr"

echo "$SEED"
for K in "${neg_ratios[@]}"
do

  echo "$exp_descr $POS al$art_len k$K s$SEED"
    #1
  python -u main.py --template train_bert_pcp --model_init_seed=$SEED --dataset_path=$data \
  --train_negative_sampler_code random --train_negative_sample_size=$K \
  --news_encoder $enc --dim_art_emb $d_art  --pt_word_emb_path=$w_emb --lower_case=1 \
  --max_article_len=$art_len --nie_layer=$nie --n_users=$n_users \
  --lr $lr --cuda_launch_blocking=1 \
  --experiment_description $exp_descr $POS al$art_len k$K s$SEED

  ((COUNTER++))
  echo "Exp counter: $COUNTER"


done