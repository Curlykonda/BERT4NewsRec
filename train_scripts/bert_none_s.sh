#!/bin/bash
#SBATCH --job-name=bertje_none
#SBATCH -n 4
#SBATCH -t 20:00:00
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:2
#SBATCH --mem=60G


module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

python --version

#srun -n 2 -t 00:30:00 --pty bash -il

# 10k users
data=("./Data/DPG_nov19/10k_time_split_n_rnd_users/")
pt_news_enc="BERTje"
pt_news_enc_path="./BertModelsPT/bert-base-dutch-cased"

SEED=$SLURM_ARRAY_TASK_ID

TE=None

art_len=30
hist_len=100

neg_ratios=(4) # 4 9

n_layers=(2 3)
n_heads=4
p_d=0.2
p_m=0.15

lr=1e-4
n_epochs=100

nie="lin_gelu"
d_model=768

n_users=10000
exp_descr="10k"
COUNTER=0
#################



for K in "${neg_ratios[@]}"
do
  for nl in "${n_layers[@]}"
  do
    echo "$exp_descr $TE al$art_len hl$hist_len k$K lr$lr L$nl H$n_heads pD$p_d pM$p_m s$SEED"
      #1
    CUDA_VISIBLE_DEVICES=0,1 python -u main.py --template train_bert_pcp --model_init_seed=$SEED --dataset_path=$data \
      --train_negative_sample_size=$K --pt_news_enc=$pt_news_enc --path_pt_news_enc=$pt_news_enc_path \
      --bert_num_blocks=$nl --bert_num_heads=$n_heads --bert_dropout=$p_d --bert_mask_prob=$p_m \
      --max_article_len=$art_len --max_hist_len=$hist_len \
      --nie_layer $nie --n_users=$n_users \
      --lr $lr --num_epochs=$n_epochs --cuda_launch_blocking=1 \
      --experiment_description $exp_descr $TE al$art_len hl$hist_len k$K lr$lr L$nl H$n_heads pD$p_d pM$p_m s$SEED

    ((COUNTER++))
    echo "Exp counter: $COUNTER"
  done
done

#--train_negative_sampler_code=rnd_brand_sens \