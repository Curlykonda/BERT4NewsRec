#!/bin/bash
#SBATCH --job-name=npa_cnn_tpe
#SBATCH -N 4
#SBATCH -t 30:00:00
#SBATCH -p gpu_shared
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

POS_EMBS=("tpe") # "lpe"
neg_ratios=(4) #

enc="wucnn"
d_art=400

n_bert_layers=1

nie="lin_gelu"
LR=(1e-3 1e-4)
n_epochs=50

n_users=100000
exp_descr="100k_NpaCNN_add"
COUNTER=0

echo "$data"
echo "$exp_descr"

echo "$SEED"
for K in "${neg_ratios[@]}"
do
  for lr in "${LR[@]}"
  do
    for POS in "${POS_EMBS[@]}"
    do

      echo "$exp_descr $POS al$art_len k$K lr$lr s$SEED" # nl$n_bert_layers
        #1
      python -u main.py --template train_bert_pcp --model_init_seed=$SEED --dataset_path=$data \
      --bert_num_blocks=$n_bert_layers --train_negative_sample_size=$K \
      --news_encoder $enc --dim_art_emb $d_art --pt_word_emb_path=$w_emb --lower_case=1 \
      --pos_embs=$POS --add_embs_func=add \
      --max_article_len=$art_len --nie_layer=$nie --n_users=$n_users \
      --lr $lr --num_epochs=$n_epochs --cuda_launch_blocking=1 \
      --experiment_description $exp_descr $POS al$art_len k$K lr$lr s$SEED

      ((COUNTER++))
      echo "Exp counter: $COUNTER"

    done
  done
done