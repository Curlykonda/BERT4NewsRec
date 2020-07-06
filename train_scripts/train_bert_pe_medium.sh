#!/bin/bash
#SBATCH --job-name=bertje_pe_m
#SBATCH -n 8
#SBATCH -t 28:00:00
#SBATCH -p gpu_shared
#SBATCH --mem=60000M


module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

python --version

#srun -n 2 -t 00:30:00 --pty bash -il

data=("./Data/DPG_nov19/40k_time_split_n_rnd_users/")
#embeddings="../embeddings/cc.nl.300.bin"
pt_news_enc="BERTje"
pt_news_enc_path="./BertModelsPT/bert-base-dutch-cased"

art_len=30
#SEEDS=(113 42)
SEED=$SLURM_ARRAY_TASK_ID
POS_EMBS=("tpe" "lpe")
neg_ratios=(4 49)

nie="lin"
#LR=(0.01, 0.001, 0.0001)
lr=0.001
#decay_step=25

n_users=40000
exp_descr="40k_rnd_com"
COUNTER=0


echo "$data"
echo "$SEED"

for POS in "${POS_EMBS[@]}"
do
  for K in "${neg_ratios[@]}"
  do
    echo "$exp_descr $POS al$art_len k$K s$SEED"

      #1
    python -u main.py --template train_bert_pcp --model_init_seed=$SEED --dataset_path=$data \
    --train_negative_sampler_code random_common --train_negative_sample_size=$K \
    --pt_news_enc=$pt_news_enc --path_pt_news_enc=$pt_news_enc_path \
    --pos_embs=$POS --max_article_len=$art_len --nie_layer $nie \
    --lr $lr --n_users=$n_users --cuda_launch_blocking=1 \
    --experiment_description $exp_descr $POS al$art_len k$K s$SEED

    ((COUNTER++))
    echo "Exp counter: $COUNTER"
  done

done

exp_descr="40k_rnd"

for POS in "${POS_EMBS[@]}"
do
  for K in "${neg_ratios[@]}"
  do
    echo "$exp_descr $POS al$art_len k$K s$SEED"
      #1
    python -u main.py --template train_bert_pcp --model_init_seed=$SEED --dataset_path=$data \
    --train_negative_sampler_code random --train_negative_sample_size=$K \
    --pt_news_enc=$pt_news_enc --path_pt_news_enc=$pt_news_enc_path \
    --pos_embs=$POS --max_article_len=$art_len --nie_layer $nie \
    --lr $lr --n_users=$n_users --cuda_launch_blocking=1 \
    --experiment_description $exp_descr $POS al$art_len k$K s$SEED

    ((COUNTER++))
    echo "Exp counter: $COUNTER"
  done

done

#echo "bert block 1"
#  #2
#python -u main.py --template train_bert_pcp --model_init_seed=$SEED \
#--pt_news_enc=$pt_news_enc --path_pt_news_enc=$pt_news_enc_path \
#--pos_embs=$POS --max_article_len=$art_len --bert_feature_method $method $N --nie_layer $nie \
#--num_epochs=100 --bert_num_blocks=1 --max_hist_len=100 \
#--lr $lr --decay_step $decay_step --cuda_launch_blocking=1 \
#--experiment_description $exp_descr $POS l$art_len s$SEED
#
#echo "hist len 50"
#  #2
#python -u main.py --template train_bert_pcp --model_init_seed=$SEED \
#--pt_news_enc=$pt_news_enc --path_pt_news_enc=$pt_news_enc_path \
#--pos_embs=$POS --max_article_len=$art_len --bert_feature_method $method $N --nie_layer $nie \
#--num_epochs=100 --bert_num_blocks=2 --max_hist_len=50 \
#--lr $lr --decay_step $decay_step --cuda_launch_blocking=1 \
#--experiment_description $exp_descr $POS l$art_len s$SEED



