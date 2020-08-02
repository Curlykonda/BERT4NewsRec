#!/bin/bash
#SBATCH --job-name=bertje_pe_l
#SBATCH -N 4
#SBATCH -t 25:00:00
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:2
#SBATCH --mem=60G

module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

python --version

#srun -n 2 -t 00:30:00 --pty bash -il

data=("./Data/DPG_nov19/100k_time_split_n_rnd_users/")
pt_news_enc="BERTje"
pt_news_enc_path="./BertModelsPT/bert-base-dutch-cased"

SEED=$SLURM_ARRAY_TASK_ID

art_len=30
hist_len=100

POS_EMBS=("lpe")
neg_ratios=(49 74 99) # 9

nie="lin_gelu"
lr=1e-4
warmup=(0)
n_epochs=50

n_users=100000
COUNTER=0
#####

exp_descr="100k_add"

for wu in "${warmup[@]}"
do
  for K in "${neg_ratios[@]}"
  do

    for POS in "${POS_EMBS[@]}"
    do
      echo "$exp_descr $POS al$art_len hl$hist_len k$K lr$lr sch s$SEED"
        #1
      CUDA_VISIBLE_DEVICES=0,1 python -u main.py --template train_bert_pcp --model_init_seed=$SEED --dataset_path=$data \
        --train_negative_sample_size=$K --pt_news_enc=$pt_news_enc --path_pt_news_enc=$pt_news_enc_path \
        --lr_schedule=1 --warmup_ratio=$wu \
        --max_article_len=$art_len --max_hist_len=$hist_len \
        --pos_embs=$POS --add_embs_func=add --nie_layer $nie \
        --lr $lr --n_users=$n_users --num_epochs=$n_epochs --cuda_launch_blocking=1 \
        --experiment_description $exp_descr $POS al$art_len k$K lr$lr sch s$SEED

      ((COUNTER++))
      echo "Exp counter: $COUNTER"
    done
  done
done