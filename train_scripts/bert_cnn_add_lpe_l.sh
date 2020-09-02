#!/bin/bash
#SBATCH --job-name=cnn_lpe_add
#SBATCH -n 4
#SBATCH -t 40:00:00
#SBATCH -p gpu_shared
#SBATCH --gres=gpu:2
#SBATCH --mem=60G

module load pre2019
module load Miniconda3/4.3.27
source activate thesis-user-modelling

python --version

#srun -n 2 -t 00:30:00 --pty bash -il

data=("./Data/DPG_nov19/100k_time_split_n_rnd_users/")
w_emb="./pc_word_embeddings/cc.nl.300.bin"

SEED=$SLURM_ARRAY_TASK_ID

art_len=30
hist_len=100

#POS_EMBS=("lpe") #
POS="lpe"
neg_ratios=(4) #

enc="wucnn"
d_art=400

n_layers=(2 3 4)
n_heads=4
p_dropout=(0.1)

nie="lin_gelu"
lr=1e-3
n_epochs=100

n_users=100000
exp_descr="100k_cnn_add"
COUNTER=0
####################################

echo "$data"

for K in "${neg_ratios[@]}"
do
  for nl in "${n_layers[@]}"
  do
    for p_d in "${p_dropout[@]}"
    do
      echo "$exp_descr $POS al$art_len hl$hist_len k$K lr$lr L$nl H$n_heads pD$p_d s$SEED"
        #1
      CUDA_VISIBLE_DEVICES=0,1 python -u main.py --template train_bert_pcp --model_init_seed=$SEED --dataset_path=$data \
        --bert_num_blocks=$nl --bert_num_heads=$n_heads --bert_dropout=$p_d \
        --train_negative_sample_size=$K \
        --pos_embs=$POS --add_embs_func=add \
        --news_encoder $enc --dim_art_emb $d_art --pt_word_emb_path=$w_emb --lower_case=1 \
        --max_article_len=$art_len --max_hist_len=$hist_len \
        --nie_layer=$nie --n_users=$n_users \
        --lr $lr --num_epochs=$n_epochs --cuda_launch_blocking=1 \
        --experiment_description $exp_descr $POS al$art_len k$K lr$lr L$nl H$n_heads pD$p_d s$SEED

      ((COUNTER++))
      echo "Exp counter: $COUNTER"
    done
  done
done