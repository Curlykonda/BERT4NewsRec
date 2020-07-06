#!/bin/bash
#SBATCH --job-name=bertje_te
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
pt_news_enc_path = "./BertModelsPT/bert-base-dutch-cased"

SEED=$SLURM_ARRAY_TASK_ID

art_len=30
neg_ratios=(4 49)
lr=0.001
#decay_step=25
TEMP_EMBS=("lte" "nte")
t_act_func="relu"

nie="lin"
d_model=768

n_users=40000
exp_descr="40k_rnd_com"
COUNTER=0

echo "$data"

for TE in "${TEMP_EMBS[@]}"
do
  for K in "${neg_ratios[@]}"
  do
    echo "$exp_descr $TE al$art_len k$K s$SEED"

      #1
    python -u main.py --template train_bert_pcp --model_init_seed=$SEED --dataset_path=$data \
    --train_negative_sampler_code random_common --train_negative_sample_size=$K \
    --pt_news_enc=$pt_news_enc --path_pt_news_enc=$pt_news_enc_path \
    --temp_embs=$TE --incl_time_stamp=1 --temp_embs_hidden_units 256 $d_art --temp_embs_act_func $t_act_func \
    --max_article_len=$art_len --nie_layer $nie --n_users=$n_users \
    --lr $lr --cuda_launch_blocking=1 \
    --experiment_description $exp_descr $TE al$art_len k$K s$SEED

    ((COUNTER++))
    echo "Exp counter: $COUNTER"
  done

done

exp_descr="40k_rnd"

for TE in "${TEMP_EMBS[@]}"
do
  for K in "${neg_ratios[@]}"
  do
    echo "$exp_descr $TE al$art_len k$K s$SEED"
      #1
    python -u main.py --template train_bert_pcp --model_init_seed=$SEED --dataset_path=$data \
    --train_negative_sampler_code random_common --train_negative_sample_size=$K \
    --pt_news_enc=$pt_news_enc --path_pt_news_enc=$pt_news_enc_path \
    --temp_embs=$TE --incl_time_stamp=1 --temp_embs_hidden_units 256 $d_art --temp_embs_act_func $t_act_func \
    --max_article_len=$art_len --nie_layer $nie --n_users=$n_users \
    --lr $lr --cuda_launch_blocking=1 \
    --experiment_description $exp_descr $TE al$art_len k$K s$SEED

    ((COUNTER++))
    echo "Exp counter: $COUNTER"
  done

done